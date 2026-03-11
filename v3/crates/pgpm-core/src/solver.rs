//! SOCP problem construction and solving.
//!
//! Paper references:
//! - Eq. 18: Overall optimization formulation
//! - Eq. 23a-c, 26: Isometric constraints
//! - Eq. 28a-b: Conformal constraints
//! - Eq. 29-30: Position constraint energy
//! - Eq. 31: Biharmonic regularization energy
//! - Eq. 33: ARAP regularization energy
//!
//! We use clarabel as the SOCP solver (replacing Mosek from Section 6).
//! Clarabel standard form:
//!   min  ½x'Px + q'x
//!   s.t. Ax + s = b,  s ∈ K
//! where K is a product of cones.

use crate::basis::BasisFunction;
use crate::distortion_policy::DistortionPolicy;
use crate::types::*;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SecondOrderConeT, SolverStatus,
};
use nalgebra::{DMatrix, DVector, Vector2};

// ─────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────

/// Solve the SOCP problem from Eq. 18.
///
/// min_c  E_pos(f) + λ E_reg(f)
/// s.t.   D(f; z) ≤ K,  ∀z ∈ Z' ∪ Z''
///        f = Σ c_i f_i
pub fn solve_socp(
    source_handles: &[Vector2<f64>],
    target_handles: &[Vector2<f64>],
    basis: &dyn BasisFunction,
    state: &AlgorithmState,
    params: &MappingParams,
    policy: &dyn DistortionPolicy,
    solver_config: &SolverConfig,
) -> Result<CoefficientMatrix, SolverError> {
    let precomputed = state.precomputed.as_ref().ok_or_else(|| {
        SolverError::NumericalError("Precomputed data not available".to_string())
    })?;

    let n_basis = basis.count();
    let n_handles = source_handles.len();
    let active_indices = collect_active_indices(state);
    let n_active = active_indices.len();

    // Decision variable layout (Eq. 18):
    // Common: [c¹(n), c²(n), r(L)]
    // Isometric adds: [t(n_active), s(n_active)]  (Eq. 23)
    let n_vars = 2 * n_basis + n_handles
        + policy.extra_vars_per_active() * n_active;

    // === Build objective (Eq. 18, 30, 31, 33) ===
    let mut q = vec![0.0; n_vars];
    // Position energy (Eq. 30): min Σ r_l
    for l in 0..n_handles {
        q[2 * n_basis + l] = 1.0;
    }
    let (p_mat, q_reg) = build_regularization(
        basis, state, params, precomputed, n_basis, n_vars,
    );
    for i in 0..q_reg.len().min(n_vars) {
        q[i] += q_reg[i];
    }

    // === Build constraints ===
    let mut rows: Vec<Vec<(usize, f64)>> = Vec::new();
    let mut b_vec: Vec<f64> = Vec::new();
    let mut cones: Vec<clarabel::solver::SupportedConeT<f64>> = Vec::new();

    // Position constraints (Eq. 30)
    append_position_constraints(
        source_handles, target_handles, basis, n_basis,
        &mut rows, &mut b_vec, &mut cones,
    );

    // Distortion constraints (Eq. 23/26 or 28) — delegated to policy
    policy.append_constraints(
        state, precomputed, n_basis, n_handles,
        &active_indices, n_active, params.k_bound,
        &mut rows, &mut b_vec, &mut cones,
    );

    // === Assemble and solve ===
    assemble_and_solve(p_mat, &q, &rows, &b_vec, &cones, n_vars, n_basis, solver_config)
}

// ─────────────────────────────────────────────
// Constraint builders
// ─────────────────────────────────────────────

/// Position constraints (Eq. 30).
///
/// ||Σ c_i f_i(p_l) - q_l|| ≤ r_l   (SOC(3) per handle)
fn append_position_constraints(
    source_handles: &[Vector2<f64>],
    target_handles: &[Vector2<f64>],
    basis: &dyn BasisFunction,
    n_basis: usize,
    rows: &mut Vec<Vec<(usize, f64)>>,
    b: &mut Vec<f64>,
    cones: &mut Vec<clarabel::solver::SupportedConeT<f64>>,
) {
    for l in 0..source_handles.len() {
        let phi_l = basis.evaluate(source_handles[l]);
        let r_col = 2 * n_basis + l;

        // s_1 = r_l
        rows.push(vec![(r_col, -1.0)]);
        b.push(0.0);

        // s_2 = q_l_x - Σ c¹_i f_i(p_l)
        let mut row = Vec::new();
        for i in 0..n_basis {
            if phi_l[i].abs() > 1e-15 {
                row.push((i, phi_l[i]));
            }
        }
        rows.push(row);
        b.push(target_handles[l].x);

        // s_3 = q_l_y - Σ c²_i f_i(p_l)
        let mut row = Vec::new();
        for i in 0..n_basis {
            if phi_l[i].abs() > 1e-15 {
                row.push((n_basis + i, phi_l[i]));
            }
        }
        rows.push(row);
        b.push(target_handles[l].y);

        cones.push(SecondOrderConeT(3));
    }
}

/// Isometric distortion constraints (Eq. 23a-c, 26).
///
/// Per active point i:
///   ||J_S f(z_i)|| ≤ t_i          SOC(3)     (Eq. 23a)
///   ||J_A f(z_i)|| ≤ s_i          SOC(3)     (Eq. 23b)
///   t_i + s_i ≤ K                  NN         (Eq. 23c)
///   J_S f(z_i)·d_i - s_i ≥ 1/K    NN         (Eq. 26)
pub(crate) fn append_isometric_constraints(
    state: &AlgorithmState,
    precomputed: &PrecomputedData,
    n_basis: usize,
    n_handles: usize,
    active_indices: &[usize],
    n_active: usize,
    k: f64,
    rows: &mut Vec<Vec<(usize, f64)>>,
    b: &mut Vec<f64>,
    cones: &mut Vec<clarabel::solver::SupportedConeT<f64>>,
) {
    for (ai, &pt_idx) in active_indices.iter().enumerate() {
        let t_col = 2 * n_basis + n_handles + ai;
        let s_col = 2 * n_basis + n_handles + n_active + ai;
        let d = state.frames[pt_idx]; // Eq. 27

        // (a) ||J_S f(z_i)|| ≤ t_i — SOC(3) (Eq. 23a)
        rows.push(vec![(t_col, -1.0)]);
        b.push(0.0);
        rows.push(j_s_x_row(precomputed, pt_idx, n_basis));
        b.push(0.0);
        rows.push(j_s_y_row(precomputed, pt_idx, n_basis));
        b.push(0.0);
        cones.push(SecondOrderConeT(3));

        // (b) ||J_A f(z_i)|| ≤ s_i — SOC(3) (Eq. 23b)
        rows.push(vec![(s_col, -1.0)]);
        b.push(0.0);
        rows.push(j_a_x_row(precomputed, pt_idx, n_basis));
        b.push(0.0);
        rows.push(j_a_y_row(precomputed, pt_idx, n_basis));
        b.push(0.0);
        cones.push(SecondOrderConeT(3));

        // (c) t_i + s_i ≤ K — NN: K - t_i - s_i ≥ 0 (Eq. 23c)
        rows.push(vec![(t_col, 1.0), (s_col, 1.0)]);
        b.push(k);

        // (d) J_S f·d_i - s_i ≥ 1/K — NN (Eq. 26)
        // Clarabel: Ax + s = b, s ≥ 0  →  Ax ≤ b
        // We want: J_S·d - s_i ≥ 1/K  →  -J_S·d + s_i ≤ -1/K
        let mut row = neg_j_s_dot_d_row(precomputed, pt_idx, n_basis, d);
        row.push((s_col, 1.0));
        rows.push(row);
        b.push(-1.0 / k);

        cones.push(NonnegativeConeT(2));
    }
}

/// Conformal distortion constraints (Eq. 28a-b).
///
/// Per active point i:
///   ||J_A f(z_i)|| ≤ ((K-1)/(K+1)) · J_S f(z_i)·d_i   (Eq. 28a)
///   ||J_A f(z_i)|| ≤ J_S f(z_i)·d_i - δ                (Eq. 28b)
pub(crate) fn append_conformal_constraints(
    state: &AlgorithmState,
    precomputed: &PrecomputedData,
    n_basis: usize,
    active_indices: &[usize],
    k: f64,
    delta: f64,
    rows: &mut Vec<Vec<(usize, f64)>>,
    b: &mut Vec<f64>,
    cones: &mut Vec<clarabel::solver::SupportedConeT<f64>>,
) {
    let ratio = (k - 1.0) / (k + 1.0);

    for &pt_idx in active_indices {
        let d = state.frames[pt_idx];

        // (a) ||J_A f(z_i)|| ≤ ratio · J_S f(z_i)·d_i — SOC(3) (Eq. 28a)
        // s_1 = ratio · J_S·d
        {
            let mut row = neg_j_s_dot_d_row(precomputed, pt_idx, n_basis, d);
            for entry in &mut row {
                entry.1 *= ratio;
            }
            rows.push(row);
            b.push(0.0);
        }
        rows.push(j_a_x_row(precomputed, pt_idx, n_basis));
        b.push(0.0);
        rows.push(j_a_y_row(precomputed, pt_idx, n_basis));
        b.push(0.0);
        cones.push(SecondOrderConeT(3));

        // (b) ||J_A f(z_i)|| ≤ J_S f(z_i)·d_i - δ — SOC(3) (Eq. 28b)
        // s_1 = J_S·d - δ = -δ - (-J_S·d terms · x)
        rows.push(neg_j_s_dot_d_row(precomputed, pt_idx, n_basis, d));
        b.push(-delta);
        rows.push(j_a_x_row(precomputed, pt_idx, n_basis));
        b.push(0.0);
        rows.push(j_a_y_row(precomputed, pt_idx, n_basis));
        b.push(0.0);
        cones.push(SecondOrderConeT(3));
    }
}

// ─────────────────────────────────────────────
// Jacobian decomposition row builders (Eq. 19)
// ─────────────────────────────────────────────

/// J_S_x(z) = (1/2)(Σ c¹_i ∂f_i/∂x + Σ c²_i ∂f_i/∂y)  (Eq. 19)
fn j_s_x_row(precomputed: &PrecomputedData, pt_idx: usize, n_basis: usize) -> Vec<(usize, f64)> {
    let mut row = Vec::new();
    for i in 0..n_basis {
        let gx = precomputed.grad_phi_x[(pt_idx, i)];
        let gy = precomputed.grad_phi_y[(pt_idx, i)];
        if gx.abs() > 1e-15 { row.push((i, 0.5 * gx)); }
        if gy.abs() > 1e-15 { row.push((n_basis + i, 0.5 * gy)); }
    }
    row
}

/// J_S_y(z) = (1/2)(Σ c¹_i ∂f_i/∂y - Σ c²_i ∂f_i/∂x)  (Eq. 19)
fn j_s_y_row(precomputed: &PrecomputedData, pt_idx: usize, n_basis: usize) -> Vec<(usize, f64)> {
    let mut row = Vec::new();
    for i in 0..n_basis {
        let gx = precomputed.grad_phi_x[(pt_idx, i)];
        let gy = precomputed.grad_phi_y[(pt_idx, i)];
        if gy.abs() > 1e-15 { row.push((i, 0.5 * gy)); }
        if gx.abs() > 1e-15 { row.push((n_basis + i, -0.5 * gx)); }
    }
    row
}

/// J_A_x(z) = (1/2)(Σ c¹_i ∂f_i/∂x - Σ c²_i ∂f_i/∂y)  (Eq. 19)
fn j_a_x_row(precomputed: &PrecomputedData, pt_idx: usize, n_basis: usize) -> Vec<(usize, f64)> {
    let mut row = Vec::new();
    for i in 0..n_basis {
        let gx = precomputed.grad_phi_x[(pt_idx, i)];
        let gy = precomputed.grad_phi_y[(pt_idx, i)];
        if gx.abs() > 1e-15 { row.push((i, 0.5 * gx)); }
        if gy.abs() > 1e-15 { row.push((n_basis + i, -0.5 * gy)); }
    }
    row
}

/// J_A_y(z) = (1/2)(Σ c¹_i ∂f_i/∂y + Σ c²_i ∂f_i/∂x)  (Eq. 19)
fn j_a_y_row(precomputed: &PrecomputedData, pt_idx: usize, n_basis: usize) -> Vec<(usize, f64)> {
    let mut row = Vec::new();
    for i in 0..n_basis {
        let gx = precomputed.grad_phi_x[(pt_idx, i)];
        let gy = precomputed.grad_phi_y[(pt_idx, i)];
        if gy.abs() > 1e-15 { row.push((i, 0.5 * gy)); }
        if gx.abs() > 1e-15 { row.push((n_basis + i, 0.5 * gx)); }
    }
    row
}

/// Negated J_S·d row for constraint matrix (Eq. 25-26).
///
/// J_S·d = J_S_x · d_x + J_S_y · d_y
/// Returns the negated coefficients: -(J_S·d) as A-matrix entries.
fn neg_j_s_dot_d_row(
    precomputed: &PrecomputedData,
    pt_idx: usize,
    n_basis: usize,
    d: Vector2<f64>,
) -> Vec<(usize, f64)> {
    let mut row = Vec::new();
    for i in 0..n_basis {
        let gx = precomputed.grad_phi_x[(pt_idx, i)];
        let gy = precomputed.grad_phi_y[(pt_idx, i)];
        // -(J_S·d) = -(1/2)(c¹(gx·d_x + gy·d_y) + c²(gy·d_x - gx·d_y))
        let coeff_c1 = -0.5 * (gx * d.x + gy * d.y);
        let coeff_c2 = -0.5 * (gy * d.x - gx * d.y);
        if coeff_c1.abs() > 1e-15 { row.push((i, coeff_c1)); }
        if coeff_c2.abs() > 1e-15 { row.push((n_basis + i, coeff_c2)); }
    }
    row
}

// ─────────────────────────────────────────────
// Solver assembly and execution
// ─────────────────────────────────────────────

/// Assemble the SOCP from components and solve with Clarabel.
///
/// Applies P matrix diagonal regularization (not part of paper, for
/// numerical stability), converts to Clarabel sparse format, and
/// extracts the coefficient matrix from the solution.
fn assemble_and_solve(
    mut p_mat: DMatrix<f64>,
    q: &[f64],
    rows: &[Vec<(usize, f64)>],
    b_vec: &[f64],
    cones: &[clarabel::solver::SupportedConeT<f64>],
    n_vars: usize,
    n_basis: usize,
    solver_config: &SolverConfig,
) -> Result<CoefficientMatrix, SolverError> {
    let (a_csc, b_arr) = sparse_rows_to_csc(rows, b_vec, n_vars);

    check_problem_data(q, &a_csc, &b_arr, &p_mat)?;

    // Diagonal regularization for numerical stability (not from the paper).
    // Prevents near-singular KKT systems when λ_reg is very small.
    for i in 0..n_vars {
        if i < 2 * n_basis {
            p_mat[(i, i)] += solver_config.p_reg_coefficient;
        } else {
            p_mat[(i, i)] += solver_config.p_reg_auxiliary;
        }
    }

    let p_csc = dense_to_csc_upper_tri(&p_mat);
    let settings = make_solver_settings();

    let mut solver = DefaultSolver::new(&p_csc, q, &a_csc, &b_arr, cones, settings);
    solver.solve();

    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => {
            let x = &solver.solution.x;
            let mut c = CoefficientMatrix::zeros(2, n_basis);
            for i in 0..n_basis {
                c[(0, i)] = x[i];
                c[(1, i)] = x[n_basis + i];
            }
            Ok(c)
        }
        SolverStatus::PrimalInfeasible | SolverStatus::DualInfeasible => {
            Err(SolverError::Infeasible(format!(
                "SOCP infeasible: {:?}",
                solver.solution.status
            )))
        }
        status => Err(SolverError::SolverFailed(format!(
            "Solver returned status: {:?}",
            status
        ))),
    }
}

// ─────────────────────────────────────────────
// Solver settings
// ─────────────────────────────────────────────

/// Build Clarabel solver settings with relaxed tolerances.
///
/// The default Clarabel tolerances (1e-8) are too tight for SOCP problems
/// constructed from pixel-coordinate-scale domains (e.g. 1000×1000 images).
/// The constraint matrix mixes basis function gradients (small values, ~1/s²)
/// with coordinate-scale position constraints, leading to poor conditioning.
///
/// Relaxing to 1e-6 / 1e-5 is standard practice for geometric optimization
/// SOCPs where moderate precision suffices.
fn make_solver_settings() -> DefaultSettings<f64> {
    DefaultSettings {
        verbose: false,
        // Relax convergence tolerances for numerical stability
        tol_gap_abs: 1e-5,
        tol_gap_rel: 1e-5,
        tol_feas: 1e-5,
        // Relax infeasibility detection so near-feasible problems
        // aren't prematurely declared infeasible
        tol_infeas_abs: 1e-5,
        tol_infeas_rel: 1e-5,
        // Relax "AlmostSolved" reduced tolerances
        reduced_tol_gap_abs: 1e-3,
        reduced_tol_gap_rel: 1e-3,
        reduced_tol_feas: 1e-2,
        // Allow more equilibration iterations for better scaling
        equilibrate_max_iter: 50,
        // Widen the equilibration scaling range to handle
        // the large scale differences in the constraint matrix
        equilibrate_min_scaling: 1e-6,
        equilibrate_max_scaling: 1e+6,
        // Increase static regularization to stabilize the KKT system
        static_regularization_enable: true,
        static_regularization_constant: 1e-7,
        // Increase iteration limit in case the relaxed tolerances
        // need more iterations to converge
        max_iter: 500,
        ..DefaultSettings::default()
    }
}

// ─────────────────────────────────────────────
// Active index collection
// ─────────────────────────────────────────────

/// Collect unique active point indices from Z' ∪ Z''.
fn collect_active_indices(state: &AlgorithmState) -> Vec<usize> {
    let mut indices: Vec<usize> = state.active_set.clone();
    for &idx in &state.stable_set {
        if !indices.contains(&idx) {
            indices.push(idx);
        }
    }
    indices.sort();
    indices
}

// ─────────────────────────────────────────────
// Regularization energy (Eq. 31, 33)
// ─────────────────────────────────────────────

/// Build the regularization quadratic form P and linear term q_reg.
///
/// Returns (P_matrix, q_linear) where:
/// - P_matrix: n_vars × n_vars quadratic form (for ½x'Px)
/// - q_linear: n_vars-length vector (for q'x, added to objective)
///
/// Only the c¹, c² portion (first 2*n_basis entries) has nonzero entries.
///
/// Regularization types:
/// - Biharmonic (Eq. 31): ∫∫ ||H_u||²_F + ||H_v||²_F dA (pure quadratic)
/// - ARAP (Eq. 33): Σ_s (||J_A f(r_s)||² + ||J_S f(r_s) - d_s||²)
///   The ARAP energy expands to quadratic + linear + constant terms.
///   The linear term (-2 d_s · J_S f) must be included in q.
fn build_regularization(
    _basis: &dyn BasisFunction,
    state: &AlgorithmState,
    params: &MappingParams,
    precomputed: &PrecomputedData,
    n_basis: usize,
    n_vars: usize,
) -> (DMatrix<f64>, Vec<f64>) {
    let lambda = params.lambda_reg;
    let mut p = DMatrix::zeros(n_vars, n_vars);
    let mut q_reg = vec![0.0; n_vars];

    if lambda.abs() < 1e-15 {
        return (p, q_reg);
    }

    let (lambda_bh, lambda_arap) = match &params.regularization {
        RegularizationType::Biharmonic => (1.0, 0.0),
        RegularizationType::Arap => (0.0, 1.0),
        RegularizationType::Mixed { lambda_bh, lambda_arap } => (*lambda_bh, *lambda_arap),
    };

    // Biharmonic energy (Eq. 31)
    if lambda_bh > 0.0 {
        if let Some(ref bh) = precomputed.biharmonic_matrix {
            // bh is (2n × 2n), add λ * λ_bh * bh to the c portion of P
            let scale = lambda * lambda_bh;
            for i in 0..2 * n_basis {
                for j in 0..2 * n_basis {
                    p[(i, j)] += scale * bh[(i, j)];
                }
            }
        }
    }

    // ARAP energy (Eq. 33)
    // E_arap = Σ_s (||J_A f(r_s)||² + ||J_S f(r_s) - d_s||²)
    //
    // Expanding ||J_S - d||²:
    //   = ||J_S||² - 2 d·J_S + ||d||²
    //   = ||J_S||² - 2(d_x·J_S_x + d_y·J_S_y) + 1
    //
    // So E_arap = Σ_s (||J_A||² + ||J_S||² - 2 d_s·J_S + 1)
    //
    // Quadratic part → P matrix: ||J_A||² + ||J_S||²
    // Linear part → q vector: -2 d_s·J_S
    // Constant part (1 per sample): ignored (doesn't affect optimization)
    //
    // Paper Eq. 33: sample points R'' are "a set of regularization sample points"
    // We use all domain-interior collocation points for stable regularization.
    // Points outside the domain contour are excluded (Section 4).
    if lambda_arap > 0.0 {
        let interior_indices: Vec<usize> = (0..state.collocation_points.len())
            .filter(|&i| state.domain_mask[i])
            .collect();
        let m_interior = interior_indices.len().max(1);
        let scale = lambda * lambda_arap / m_interior as f64; // normalize by interior count

        for &pt_idx in &interior_indices {
            let d = state.frames[pt_idx];

            let mut a_js_x = DVector::zeros(2 * n_basis);
            let mut a_js_y = DVector::zeros(2 * n_basis);
            let mut a_ja_x = DVector::zeros(2 * n_basis);
            let mut a_ja_y = DVector::zeros(2 * n_basis);

            for i in 0..n_basis {
                let gx = precomputed.grad_phi_x[(pt_idx, i)];
                let gy = precomputed.grad_phi_y[(pt_idx, i)];

                // J_S_x = (1/2)(c¹ gx + c² gy)
                a_js_x[i] = 0.5 * gx;
                a_js_x[n_basis + i] = 0.5 * gy;

                // J_S_y = (1/2)(c¹ gy - c² gx)
                a_js_y[i] = 0.5 * gy;
                a_js_y[n_basis + i] = -0.5 * gx;

                // J_A_x = (1/2)(c¹ gx - c² gy)
                a_ja_x[i] = 0.5 * gx;
                a_ja_x[n_basis + i] = -0.5 * gy;

                // J_A_y = (1/2)(c¹ gy + c² gx)
                a_ja_y[i] = 0.5 * gy;
                a_ja_y[n_basis + i] = 0.5 * gx;
            }

            // Quadratic part: ||J_A||² + ||J_S||²
            for i in 0..2 * n_basis {
                for j in 0..2 * n_basis {
                    p[(i, j)] += scale * (
                        a_ja_x[i] * a_ja_x[j] + a_ja_y[i] * a_ja_y[j]
                        + a_js_x[i] * a_js_x[j] + a_js_y[i] * a_js_y[j]
                    );
                }
            }

            // Linear part: -2 d·J_S = -2(d_x·J_S_x + d_y·J_S_y)
            // J_S_x = a_js_x · c, J_S_y = a_js_y · c
            // → q contribution: -2 * (d_x * a_js_x + d_y * a_js_y)
            for i in 0..2 * n_basis {
                q_reg[i] += scale * (-2.0) * (d.x * a_js_x[i] + d.y * a_js_y[i]);
            }
        }
    }

    (p, q_reg)
}

/// Build biharmonic energy matrix (Eq. 31) by numerical integration.
///
/// E_bh = ∫∫_Ω (||H_u||²_F + ||H_v||²_F) dA
///
/// where H_u is the Hessian of u, ||H||²_F = h_xx² + 2*h_xy² + h_yy².
///
/// This is a quadratic form in c:
/// E_bh = c^T M_bh c
/// where M_bh is computed by numerically integrating over the domain.
pub fn build_biharmonic_matrix(
    basis: &dyn BasisFunction,
    collocation_points: &[Vector2<f64>],
    domain_bounds: &DomainBounds,
    n_basis: usize,
) -> DMatrix<f64> {
    let m = collocation_points.len();
    let dim = 2 * n_basis;
    let mut mat = DMatrix::zeros(dim, dim);

    // Use collocation points as quadrature points with equal weights.
    // Weight = domain area / number of points (simple quadrature)
    let area =
        (domain_bounds.x_max - domain_bounds.x_min) * (domain_bounds.y_max - domain_bounds.y_min);
    let weight = area / m as f64;

    for pt in collocation_points {
        let (hxx, hxy, hyy) = basis.hessian(*pt);

        // For u component (c¹): ||H_u||²_F = (Σ c¹_i h_xx_i)² + 2(Σ c¹_i h_xy_i)² + (Σ c¹_i h_yy_i)²
        // This is a quadratic form: c¹^T (h_xx h_xx^T + 2 h_xy h_xy^T + h_yy h_yy^T) c¹
        // Same for v component (c²) in the lower-right block.

        for i in 0..n_basis {
            for j in 0..n_basis {
                let q = weight * (hxx[i] * hxx[j] + 2.0 * hxy[i] * hxy[j] + hyy[i] * hyy[j]);
                // u-block (top-left)
                mat[(i, j)] += q;
                // v-block (bottom-right)
                mat[(n_basis + i, n_basis + j)] += q;
            }
        }
    }

    mat
}

// ─────────────────────────────────────────────
// Utility: sparse matrix construction for clarabel
// ─────────────────────────────────────────────

/// Convert sparse row representation to CscMatrix for clarabel.
fn sparse_rows_to_csc(
    rows: &[Vec<(usize, f64)>],
    b: &[f64],
    n_cols: usize,
) -> (CscMatrix<f64>, Vec<f64>) {
    let n_rows = rows.len();

    // Count entries per column
    let mut col_counts = vec![0usize; n_cols];
    for row in rows {
        for &(col, _) in row {
            col_counts[col] += 1;
        }
    }

    // Build colptr
    let mut colptr = vec![0usize; n_cols + 1];
    for j in 0..n_cols {
        colptr[j + 1] = colptr[j] + col_counts[j];
    }
    let nnz = colptr[n_cols];

    // Fill rowval and nzval
    let mut rowval = vec![0usize; nnz];
    let mut nzval = vec![0.0f64; nnz];
    let mut col_pos = vec![0usize; n_cols]; // current insert position per column

    for (row_idx, row) in rows.iter().enumerate() {
        for &(col, val) in row {
            let pos = colptr[col] + col_pos[col];
            rowval[pos] = row_idx;
            nzval[pos] = val;
            col_pos[col] += 1;
        }
    }

    // Sort entries within each column by row index (required by CscMatrix)
    for j in 0..n_cols {
        let start = colptr[j];
        let end = colptr[j + 1];
        if end > start {
            let mut pairs: Vec<(usize, f64)> = (start..end)
                .map(|k| (rowval[k], nzval[k]))
                .collect();
            pairs.sort_by_key(|&(r, _)| r);
            for (k, (r, v)) in pairs.into_iter().enumerate() {
                rowval[start + k] = r;
                nzval[start + k] = v;
            }
        }
    }

    let a = CscMatrix::new(n_rows, n_cols, colptr, rowval, nzval);
    (a, b.to_vec())
}

/// Convert a dense symmetric matrix to upper-triangular CscMatrix.
/// Clarabel requires P to be upper triangular.
fn dense_to_csc_upper_tri(mat: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = mat.nrows();
    assert_eq!(n, mat.ncols());

    let mut colptr = vec![0usize; n + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();

    for j in 0..n {
        for i in 0..=j {
            let val = mat[(i, j)];
            if val.abs() > 1e-15 {
                rowval.push(i);
                nzval.push(val);
            }
        }
        colptr[j + 1] = rowval.len();
    }

    CscMatrix::new(n, n, colptr, rowval, nzval)
}

/// Check problem data for NaN/Inf values that would corrupt the solver.
fn check_problem_data(
    q: &[f64],
    a_csc: &CscMatrix<f64>,
    b: &[f64],
    p: &DMatrix<f64>,
) -> Result<(), SolverError> {
    for (i, &val) in q.iter().enumerate() {
        if !val.is_finite() {
            return Err(SolverError::NumericalError(
                format!("NaN/Inf in objective q[{}] = {}", i, val),
            ));
        }
    }
    for (i, &val) in b.iter().enumerate() {
        if !val.is_finite() {
            return Err(SolverError::NumericalError(
                format!("NaN/Inf in constraint b[{}] = {}", i, val),
            ));
        }
    }
    for (i, &val) in a_csc.nzval.iter().enumerate() {
        if !val.is_finite() {
            return Err(SolverError::NumericalError(
                format!("NaN/Inf in constraint matrix A nzval[{}] = {}", i, val),
            ));
        }
    }
    for i in 0..p.nrows() {
        for j in 0..p.ncols() {
            if !p[(i, j)].is_finite() {
                return Err(SolverError::NumericalError(
                    format!("NaN/Inf in P[{}, {}] = {}", i, j, p[(i, j)]),
                ));
            }
        }
    }
    Ok(())
}
