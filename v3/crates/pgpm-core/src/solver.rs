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
use crate::types::*;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SecondOrderConeT, SolverStatus,
};
use nalgebra::{DMatrix, DVector, Vector2};

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
    params: &AlgorithmParams,
) -> Result<CoefficientMatrix, SolverError> {
    let precomputed = state.precomputed.as_ref().ok_or_else(|| {
        SolverError::NumericalError("Precomputed data not available".to_string())
    })?;

    let n_basis = basis.count();
    // Decision variables layout:
    // [c¹_0, c¹_1, ..., c¹_{n-1}, c²_0, c²_1, ..., c²_{n-1}, r_0, ..., r_{L-1}, t_0, s_0, t_1, s_1, ...]
    //  \___________ 2n ___________/  \_______ L _______/  \_______ 2*n_active ________/
    //
    // where:
    //   - c¹, c² are the coefficient vectors (2n variables)
    //   - r_l are position constraint epigraph variables (L = num handles)
    //   - t_i, s_i are auxiliary variables for isometric constraints
    //     (one pair per active collocation point)

    let n_handles = source_handles.len();

    // Collect active point indices (Z' ∪ Z'')
    let active_indices = collect_active_indices(state);
    let n_active = active_indices.len();

    match params.distortion_type {
        DistortionType::Isometric => {
            build_and_solve_isometric(
                source_handles,
                target_handles,
                basis,
                state,
                params,
                precomputed,
                n_basis,
                n_handles,
                &active_indices,
                n_active,
            )
        }
        DistortionType::Conformal { delta } => {
            build_and_solve_conformal(
                source_handles,
                target_handles,
                basis,
                state,
                params,
                precomputed,
                n_basis,
                n_handles,
                &active_indices,
                n_active,
                delta,
            )
        }
    }
}

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

/// Build and solve the isometric distortion SOCP.
///
/// Decision variables: [c¹ (n), c² (n), r (L), t (n_active), s (n_active)]
/// Total: 2n + L + 2*n_active
///
/// Objective (Eq. 18, 30, 31, 33):
///   min  Σ_l r_l  +  λ · E_reg(c)
///   = min q'x + ½ x'Px
///
/// Isometric constraints per active point i (Eq. 23, 26):
///   ||J_S f(z_i)|| ≤ t_i          SOC(3)     (Eq. 23a)
///   ||J_A f(z_i)|| ≤ s_i          SOC(3)     (Eq. 23b)
///   t_i + s_i ≤ K                  NN         (Eq. 23c)
///   J_S f(z_i)·d_i - s_i ≥ 1/K    NN         (Eq. 26)
///
/// Position constraints per handle l (Eq. 30):
///   ||Σ c_i f_i(p_l) - q_l|| ≤ r_l   SOC(3)
fn build_and_solve_isometric(
    source_handles: &[Vector2<f64>],
    target_handles: &[Vector2<f64>],
    basis: &dyn BasisFunction,
    state: &AlgorithmState,
    params: &AlgorithmParams,
    precomputed: &PrecomputedData,
    n_basis: usize,
    n_handles: usize,
    active_indices: &[usize],
    n_active: usize,
) -> Result<CoefficientMatrix, SolverError> {
    let k = params.k_bound;
    let n_vars = 2 * n_basis + n_handles + 2 * n_active;

    // === Build objective ===
    // q: linear objective
    let mut q = vec![0.0; n_vars];
    // Position energy: min Σ r_l → coefficients of r_l = 1
    for l in 0..n_handles {
        q[2 * n_basis + l] = 1.0;
    }

    // P: quadratic objective (regularization energy)
    // q_reg: linear objective from ARAP regularization
    let (p_mat, q_reg) = build_regularization(
        basis, state, params, precomputed, n_basis, n_vars,
    );

    // Add ARAP linear term to q
    for i in 0..q_reg.len().min(n_vars) {
        q[i] += q_reg[i];
    }

    // === Build constraints ===
    // We'll collect constraint rows, then stack them.
    let mut constraint_rows: Vec<Vec<(usize, f64)>> = Vec::new(); // sparse rows of A
    let mut b_vec: Vec<f64> = Vec::new();
    let mut cones: Vec<clarabel::solver::SupportedConeT<f64>> = Vec::new();

    // --- Position constraints (Eq. 30): ||f(p_l) - q_l|| ≤ r_l ---
    // SOC(3): [r_l; f_u(p_l) - q_l_x; f_v(p_l) - q_l_y]
    // In clarabel form: ||s_{2..}|| ≤ s_1 where Ax + s = b
    // We need: s = b - Ax ∈ SOC(3)
    // s_1 = r_l  →  -r_l + s_1 = 0  →  A row has -1 at r_l col
    // s_2 = q_l_x - f_u(p_l) = q_l_x - Σ c¹_i f_i(p_l)
    //   → A row has f_i(p_l) at c¹_i cols, b = q_l_x
    // s_3 = q_l_y - f_v(p_l) = q_l_y - Σ c²_i f_i(p_l)
    //   → A row has f_i(p_l) at c²_i cols, b = q_l_y
    for l in 0..n_handles {
        let phi_l = basis.evaluate(source_handles[l]);
        let r_col = 2 * n_basis + l;

        // Row for s_1 = r_l: A has -1 at r_col, b = 0
        let mut row = Vec::new();
        row.push((r_col, -1.0));
        constraint_rows.push(row);
        b_vec.push(0.0);

        // Row for s_2 = q_l_x - Σ c¹_i f_i(p_l)
        let mut row = Vec::new();
        for i in 0..n_basis {
            if phi_l[i].abs() > 1e-15 {
                row.push((i, phi_l[i])); // c¹_i column
            }
        }
        constraint_rows.push(row);
        b_vec.push(target_handles[l].x);

        // Row for s_3 = q_l_y - Σ c²_i f_i(p_l)
        let mut row = Vec::new();
        for i in 0..n_basis {
            if phi_l[i].abs() > 1e-15 {
                row.push((n_basis + i, phi_l[i])); // c²_i column
            }
        }
        constraint_rows.push(row);
        b_vec.push(target_handles[l].y);

        cones.push(SecondOrderConeT(3));
    }

    // --- Isometric constraints per active point ---
    for (ai, &pt_idx) in active_indices.iter().enumerate() {
        let t_col = 2 * n_basis + n_handles + ai;
        let s_col = 2 * n_basis + n_handles + n_active + ai;
        let d = state.frames[pt_idx]; // frame vector d_i (Eq. 27)

        // J_S f(z) and J_A f(z) are linear in c (I = CW π/2 rotation):
        // J_S f = (1/2)[ Σ c¹_i ∂f_i/∂x + Σ c²_i ∂f_i/∂y,
        //                Σ c¹_i ∂f_i/∂y - Σ c²_i ∂f_i/∂x ]
        // J_A f = (1/2)[ Σ c¹_i ∂f_i/∂x - Σ c²_i ∂f_i/∂y,
        //                Σ c¹_i ∂f_i/∂y + Σ c²_i ∂f_i/∂x ]

        // (a) ||J_S f(z_i)|| ≤ t_i — SOC(3)
        // s = [t_i; -J_S_x; -J_S_y] ∈ SOC(3)
        // Row for t_i
        {
            let mut row = Vec::new();
            row.push((t_col, -1.0));
            constraint_rows.push(row);
            b_vec.push(0.0);
        }
        // Row for J_S_x
        {
            let mut row = Vec::new();
            for i in 0..n_basis {
                let gx = precomputed.grad_phi_x[(pt_idx, i)];
                let gy = precomputed.grad_phi_y[(pt_idx, i)];
                // J_S_x = (1/2)(Σ c¹_i gx_i + Σ c²_i gy_i)
                if gx.abs() > 1e-15 {
                    row.push((i, 0.5 * gx));           // c¹_i
                }
                if gy.abs() > 1e-15 {
                    row.push((n_basis + i, 0.5 * gy)); // c²_i
                }
            }
            constraint_rows.push(row);
            b_vec.push(0.0);
        }
        // Row for J_S_y
        {
            let mut row = Vec::new();
            for i in 0..n_basis {
                let gx = precomputed.grad_phi_x[(pt_idx, i)];
                let gy = precomputed.grad_phi_y[(pt_idx, i)];
                // J_S_y = (1/2)(Σ c¹_i gy_i - Σ c²_i gx_i)
                if gy.abs() > 1e-15 {
                    row.push((i, 0.5 * gy));           // c¹_i
                }
                if gx.abs() > 1e-15 {
                    row.push((n_basis + i, -0.5 * gx)); // c²_i
                }
            }
            constraint_rows.push(row);
            b_vec.push(0.0);
        }
        cones.push(SecondOrderConeT(3));

        // (b) ||J_A f(z_i)|| ≤ s_i — SOC(3)
        {
            let mut row = Vec::new();
            row.push((s_col, -1.0));
            constraint_rows.push(row);
            b_vec.push(0.0);
        }
        // Row for J_A_x
        {
            let mut row = Vec::new();
            for i in 0..n_basis {
                let gx = precomputed.grad_phi_x[(pt_idx, i)];
                let gy = precomputed.grad_phi_y[(pt_idx, i)];
                // J_A_x = (1/2)(Σ c¹_i gx_i - Σ c²_i gy_i)
                if gx.abs() > 1e-15 {
                    row.push((i, 0.5 * gx));           // c¹_i
                }
                if gy.abs() > 1e-15 {
                    row.push((n_basis + i, -0.5 * gy)); // c²_i
                }
            }
            constraint_rows.push(row);
            b_vec.push(0.0);
        }
        // Row for J_A_y
        {
            let mut row = Vec::new();
            for i in 0..n_basis {
                let gx = precomputed.grad_phi_x[(pt_idx, i)];
                let gy = precomputed.grad_phi_y[(pt_idx, i)];
                // J_A_y = (1/2)(Σ c¹_i gy_i + Σ c²_i gx_i)
                if gy.abs() > 1e-15 {
                    row.push((i, 0.5 * gy));          // c¹_i
                }
                if gx.abs() > 1e-15 {
                    row.push((n_basis + i, 0.5 * gx)); // c²_i
                }
            }
            constraint_rows.push(row);
            b_vec.push(0.0);
        }
        cones.push(SecondOrderConeT(3));

        // (c) t_i + s_i ≤ K — Nonnegative constraint: K - t_i - s_i ≥ 0
        {
            let mut row = Vec::new();
            row.push((t_col, 1.0));
            row.push((s_col, 1.0));
            constraint_rows.push(row);
            b_vec.push(k);
        }
        // (d) J_S f(z_i)·d_i - s_i ≥ 1/K — Nonnegative: J_S·d - s - 1/K ≥ 0
        {
            let mut row = Vec::new();
            for i in 0..n_basis {
                let gx = precomputed.grad_phi_x[(pt_idx, i)];
                let gy = precomputed.grad_phi_y[(pt_idx, i)];
                // J_S·d = J_S_x * d_x + J_S_y * d_y
                // J_S_x = (1/2)(c¹ gx + c² gy)
                // J_S_y = (1/2)(c¹ gy - c² gx)
                // J_S·d = (1/2)(c¹_i gx d_x + c¹_i gy d_y) + (1/2)(c²_i gy d_x - c²_i gx d_y)
                let coeff_c1 = -0.5 * (gx * d.x + gy * d.y);
                let coeff_c2 = -0.5 * (gy * d.x - gx * d.y);
                if coeff_c1.abs() > 1e-15 {
                    row.push((i, coeff_c1));
                }
                if coeff_c2.abs() > 1e-15 {
                    row.push((n_basis + i, coeff_c2));
                }
            }
            row.push((s_col, 1.0)); // -(-s_i) = +s_i on A side; we want J_S·d - s ≥ 1/K
            // In clarabel: Ax + s_slack = b, s_slack ∈ NN
            // We want J_S·d - s_i ≥ 1/K → -J_S·d + s_i + s_slack = -1/K ... no
            // Let me redo this carefully:
            // Clarabel: Ax + s = b, s ≥ 0 (NonnegativeCone)
            // We want: J_S f·d - s_i - 1/K ≥ 0
            // Let slack = J_S f·d - s_i - 1/K ≥ 0
            // → slack = -A_row · x + b_val where b_val comes from constant terms
            // A_row · x = -(coefficients of J_S·d) + s_i
            // b_val = 1/K
            // → A: negate the J_S·d coefficients, +1 at s_col
            // → b: 1/K
            // Since we already put negative of J_S·d coefficients above (coeff_c1, coeff_c2),
            // and +1 at s_col, this is:
            // Ax = (-J_S·d coefficients for c) + s_i
            // s_slack = b - Ax = 1/K - (-J_S·d + s_i) = 1/K + J_S·d - s_i
            // We need s_slack ≥ 0, so J_S·d - s_i ≥ -1/K... that's not right.

            // Let me reconsider. We want: J_S·d - s_i ≥ 1/K
            // → -(−J_S·d + s_i) ≥ 1/K
            // → -J_S·d + s_i ≤ -1/K
            // → -J_S·d + s_i + s_slack = -1/K with s_slack ≤ 0... no.
            //
            // Clarabel NonnegativeCone: s ≥ 0
            // Ax + s = b → s = b - Ax ≥ 0 → Ax ≤ b
            //
            // We want: J_S f·d - s_i ≥ 1/K
            // Negate: -(J_S f·d - s_i) ≤ -(1/K)
            // → -J_S f·d + s_i ≤ -1/K
            // But -1/K < 0, and Ax ≤ b with b < 0... this is fine for clarabel.
            // A row: (-coeff of J_S·d for c¹_i, -coeff of J_S·d for c²_i, +1 at s_col)
            // b = -1/K
            // Then s = b - Ax = -1/K - A·x ≥ 0
            // → A·x ≤ -1/K
            // → -J_S·d + s_i ≤ -1/K
            // → J_S·d - s_i ≥ 1/K  ✓
            constraint_rows.push(row);
            b_vec.push(-1.0 / k);
        }
        // Two NN constraints (Eq. 23c and Eq. 26)
        cones.push(NonnegativeConeT(2));
    }

    // Convert to clarabel format and solve
    let (a_csc, b_arr) = sparse_rows_to_csc(&constraint_rows, &b_vec, n_vars);

    // P matrix (upper triangular)
    let p_csc = dense_to_csc_upper_tri(&p_mat);

    let settings = DefaultSettings {
        verbose: false,
        ..DefaultSettings::default()
    };

    let mut solver = DefaultSolver::new(&p_csc, &q, &a_csc, &b_arr, &cones, settings);

    solver.solve();

    match solver.solution.status {
        SolverStatus::Solved | SolverStatus::AlmostSolved => {
            // Extract coefficients
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

/// Build and solve the conformal distortion SOCP.
///
/// Conformal constraints per active point (Eq. 28):
///   ||J_A f(z_i)|| ≤ ((K-1)/(K+1)) · J_S f(z_i)·d_i    (Eq. 28a)
///   ||J_A f(z_i)|| ≤ J_S f(z_i)·d_i - δ                 (Eq. 28b)
fn build_and_solve_conformal(
    source_handles: &[Vector2<f64>],
    target_handles: &[Vector2<f64>],
    basis: &dyn BasisFunction,
    state: &AlgorithmState,
    params: &AlgorithmParams,
    precomputed: &PrecomputedData,
    n_basis: usize,
    n_handles: usize,
    active_indices: &[usize],
    _n_active: usize,
    delta: f64,
) -> Result<CoefficientMatrix, SolverError> {
    let k = params.k_bound;
    // No auxiliary t, s variables for conformal (the constraints are directly SOC)
    let n_vars = 2 * n_basis + n_handles;

    // === Build objective ===
    let mut q = vec![0.0; n_vars];
    for l in 0..n_handles {
        q[2 * n_basis + l] = 1.0;
    }

    let (p_mat, q_reg) = build_regularization(
        basis, state, params, precomputed, n_basis, n_vars,
    );

    // Add ARAP linear term to q
    for i in 0..q_reg.len().min(n_vars) {
        q[i] += q_reg[i];
    }

    // === Build constraints ===
    let mut constraint_rows: Vec<Vec<(usize, f64)>> = Vec::new();
    let mut b_vec: Vec<f64> = Vec::new();
    let mut cones: Vec<clarabel::solver::SupportedConeT<f64>> = Vec::new();

    // Position constraints (same as isometric)
    for l in 0..n_handles {
        let phi_l = basis.evaluate(source_handles[l]);
        let r_col = 2 * n_basis + l;

        let mut row = Vec::new();
        row.push((r_col, -1.0));
        constraint_rows.push(row);
        b_vec.push(0.0);

        let mut row = Vec::new();
        for i in 0..n_basis {
            if phi_l[i].abs() > 1e-15 {
                row.push((i, phi_l[i]));
            }
        }
        constraint_rows.push(row);
        b_vec.push(target_handles[l].x);

        let mut row = Vec::new();
        for i in 0..n_basis {
            if phi_l[i].abs() > 1e-15 {
                row.push((n_basis + i, phi_l[i]));
            }
        }
        constraint_rows.push(row);
        b_vec.push(target_handles[l].y);

        cones.push(SecondOrderConeT(3));
    }

    // Conformal constraints per active point
    let ratio = (k - 1.0) / (k + 1.0);

    for &pt_idx in active_indices {
        let d = state.frames[pt_idx];

        // Helper: compute J_S f·d coefficient for a given basis function
        // J_S·d = (1/2)(gx*d_x + gy*d_y) for c¹ component
        //       + (1/2)(gy*d_x - gx*d_y) for c² component

        // (a) ||J_A f(z_i)|| ≤ ratio · J_S f(z_i)·d_i — SOC(3)
        // Rewrite: ||J_A|| ≤ ratio * (J_S·d)
        // SOC form: [ratio*(J_S·d); J_A_x; J_A_y]
        // s = b - Ax, s ∈ SOC(3), so s_1 = ratio*(J_S·d), s_2,3 = -J_A
        {
            // s_1 = ratio * J_S·d
            let mut row = Vec::new();
            for i in 0..n_basis {
                let gx = precomputed.grad_phi_x[(pt_idx, i)];
                let gy = precomputed.grad_phi_y[(pt_idx, i)];
                let coeff_c1 = -ratio * 0.5 * (gx * d.x + gy * d.y);
                let coeff_c2 = -ratio * 0.5 * (gy * d.x - gx * d.y);
                if coeff_c1.abs() > 1e-15 {
                    row.push((i, coeff_c1));
                }
                if coeff_c2.abs() > 1e-15 {
                    row.push((n_basis + i, coeff_c2));
                }
            }
            constraint_rows.push(row);
            b_vec.push(0.0);

            // s_2, s_3 = -J_A (same layout as isometric J_A)
            // J_A_x = (1/2)(c¹ gx - c² gy)
            let mut row = Vec::new();
            for i in 0..n_basis {
                let gx = precomputed.grad_phi_x[(pt_idx, i)];
                let gy = precomputed.grad_phi_y[(pt_idx, i)];
                if gx.abs() > 1e-15 {
                    row.push((i, 0.5 * gx));
                }
                if gy.abs() > 1e-15 {
                    row.push((n_basis + i, -0.5 * gy));
                }
            }
            constraint_rows.push(row);
            b_vec.push(0.0);

            // J_A_y = (1/2)(c¹ gy + c² gx)
            let mut row = Vec::new();
            for i in 0..n_basis {
                let gx = precomputed.grad_phi_x[(pt_idx, i)];
                let gy = precomputed.grad_phi_y[(pt_idx, i)];
                if gy.abs() > 1e-15 {
                    row.push((i, 0.5 * gy));
                }
                if gx.abs() > 1e-15 {
                    row.push((n_basis + i, 0.5 * gx));
                }
            }
            constraint_rows.push(row);
            b_vec.push(0.0);

            cones.push(SecondOrderConeT(3));
        }

        // (b) ||J_A f(z_i)|| ≤ J_S f(z_i)·d_i - δ — SOC(3)
        // s_1 = J_S·d - δ
        {
            let mut row = Vec::new();
            for i in 0..n_basis {
                let gx = precomputed.grad_phi_x[(pt_idx, i)];
                let gy = precomputed.grad_phi_y[(pt_idx, i)];
                let coeff_c1 = -0.5 * (gx * d.x + gy * d.y);
                let coeff_c2 = -0.5 * (gy * d.x - gx * d.y);
                if coeff_c1.abs() > 1e-15 {
                    row.push((i, coeff_c1));
                }
                if coeff_c2.abs() > 1e-15 {
                    row.push((n_basis + i, coeff_c2));
                }
            }
            constraint_rows.push(row);
            b_vec.push(-delta); // b - Ax → s_1 = -delta - (-J_S·d) = J_S·d - delta

            // J_A components (same as above)
            // J_A_x = (1/2)(c¹ gx - c² gy)
            let mut row = Vec::new();
            for i in 0..n_basis {
                let gx = precomputed.grad_phi_x[(pt_idx, i)];
                let gy = precomputed.grad_phi_y[(pt_idx, i)];
                if gx.abs() > 1e-15 {
                    row.push((i, 0.5 * gx));
                }
                if gy.abs() > 1e-15 {
                    row.push((n_basis + i, -0.5 * gy));
                }
            }
            constraint_rows.push(row);
            b_vec.push(0.0);

            // J_A_y = (1/2)(c¹ gy + c² gx)
            let mut row = Vec::new();
            for i in 0..n_basis {
                let gx = precomputed.grad_phi_x[(pt_idx, i)];
                let gy = precomputed.grad_phi_y[(pt_idx, i)];
                if gy.abs() > 1e-15 {
                    row.push((i, 0.5 * gy));
                }
                if gx.abs() > 1e-15 {
                    row.push((n_basis + i, 0.5 * gx));
                }
            }
            constraint_rows.push(row);
            b_vec.push(0.0);

            cones.push(SecondOrderConeT(3));
        }
    }

    // Convert and solve
    let (a_csc, b_arr) = sparse_rows_to_csc(&constraint_rows, &b_vec, n_vars);
    let p_csc = dense_to_csc_upper_tri(&p_mat);

    let settings = DefaultSettings {
        verbose: false,
        ..DefaultSettings::default()
    };

    let mut solver = DefaultSolver::new(&p_csc, &q, &a_csc, &b_arr, &cones, settings);

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
    params: &AlgorithmParams,
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
    // We use ALL collocation points for stable regularization.
    if lambda_arap > 0.0 {
        let m = state.collocation_points.len();
        let scale = lambda * lambda_arap / m as f64; // normalize by count

        for pt_idx in 0..m {
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
