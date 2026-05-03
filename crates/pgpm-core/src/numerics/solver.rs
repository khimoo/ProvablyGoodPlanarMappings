//! SOCP問題の構築と求解。
//!
//! 論文の参照:
//! - Eq. 18: 全体の最適化定式化
//! - Eq. 23a-c, 26: 等長制約
//! - Eq. 28a-b: 等角制約
//! - Eq. 29-30: 位置拘束エネルギー
//! - Eq. 31: Biharmonic正則化エネルギー
//! - Eq. 33: ARAP正則化エネルギー
//!
//! SOCPソルバーとして clarabel を使用（Section 6 の Mosek を置換）。
//! Clarabel の標準形:
//!   min  ½x'Px + q'x
//!   s.t. Ax + s = b,  s ∈ K
//! ここで K は錐の直積。

use crate::basis::BasisFunction;
use crate::model::types::*;
use crate::policy::DistortionPolicy;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, DefaultSolver, IPSolver, NonnegativeConeT, SecondOrderConeT, SolverStatus,
};
use nalgebra::{DMatrix, DVector, Vector2};

// ─────────────────────────────────────────────
// 公開API
// ─────────────────────────────────────────────

/// Eq. 18 のSOCP問題を求解する。
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

    // 決定変数のレイアウト (Eq. 18):
    // 共通: [c¹(n), c²(n), r(L)]
    // 等長の場合に追加: [t(n_active), s(n_active)]  (Eq. 23)
    let n_vars = 2 * n_basis + n_handles
        + policy.extra_vars_per_active() * n_active;

    // === 目的関数の構築 (Eq. 18, 30, 31, 33) ===
    let mut q = vec![0.0; n_vars];
    // 位置エネルギー (Eq. 30): min Σ r_l
    for l in 0..n_handles {
        q[2 * n_basis + l] = 1.0;
    }
    let (p_mat, q_reg) = build_regularization(
        basis, state, params, precomputed, n_basis, n_vars,
    );
    for i in 0..q_reg.len().min(n_vars) {
        q[i] += q_reg[i];
    }

    // === 制約の構築 ===
    let mut rows: Vec<Vec<(usize, f64)>> = Vec::new();
    let mut b_vec: Vec<f64> = Vec::new();
    let mut cones: Vec<clarabel::solver::SupportedConeT<f64>> = Vec::new();

    // 位置制約 (Eq. 30)
    append_position_constraints(
        source_handles, target_handles, basis, n_basis,
        &mut rows, &mut b_vec, &mut cones,
    );

    // 歪み制約 (Eq. 23/26 または 28) — ポリシーに委譲
    policy.append_constraints(
        state, precomputed, n_basis, n_handles,
        &active_indices, n_active, params.k_bound,
        &mut rows, &mut b_vec, &mut cones,
    );

    // === 組み立てと求解 ===
    assemble_and_solve(p_mat, &q, &rows, &b_vec, &cones, n_vars, n_basis, solver_config)
}

// ─────────────────────────────────────────────
// 制約ビルダー
// ─────────────────────────────────────────────

/// 位置制約 (Eq. 30)。
///
/// ||Σ c_i f_i(p_l) - q_l|| ≤ r_l   (ハンドルごとに SOC(3))
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

/// 等長歪み制約 (Eq. 23a-c, 26)。
///
/// アクティブ点 i ごとに:
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
        // J_S·d - s_i ≥ 1/K  →  -J_S·d + s_i ≤ -1/K
        let mut row = neg_j_s_dot_d_row(precomputed, pt_idx, n_basis, d);
        row.push((s_col, 1.0));
        rows.push(row);
        b.push(-1.0 / k);

        cones.push(NonnegativeConeT(2));
    }
}

/// 等角歪み制約 (Eq. 28a-b)。
///
/// アクティブ点 i ごとに:
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
        // s_1 = J_S·d - δ = -δ - (-J_S·d項 · x)
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
// ヤコビアン分解の行ビルダー (Eq. 19)
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

/// 制約行列用の符号反転 J_S·d 行 (Eq. 25-26)。
///
/// J_S·d = J_S_x · d_x + J_S_y · d_y
/// 符号反転した係数: -(J_S·d) をA行列のエントリとして返す。
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
// ソルバーの組み立てと実行
// ─────────────────────────────────────────────

/// 構成要素からSOCPを組み立て、Clarabelで求解する。
///
/// P行列の対角正則化（論文由来ではなく、数値安定性のため）を適用し、
/// Clarabelの疎形式に変換し、解から係数行列を抽出する。
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

    // 数値安定性のための対角正則化（論文由来ではない）。
    // λ_reg が非常に小さい場合のKKTシステムの特異近傍を防止する。
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
// ソルバー設定
// ─────────────────────────────────────────────

/// 緩和された許容誤差でClarabelソルバー設定を構築する。
///
/// Clarabelのデフォルト許容誤差 (1e-8) は、ピクセル座標スケールの
/// ドメイン（例: 1000×1000画像）から構築されたSOCP問題には厳しすぎる。
/// 制約行列が基底関数の勾配（小さい値、~1/s²）と座標スケールの
/// 位置制約を混在させるため、条件数が悪化する。
///
/// 1e-6 / 1e-5 への緩和は、中程度の精度で十分な
/// 幾何最適化SOCPの標準的手法。
fn make_solver_settings() -> DefaultSettings<f64> {
    DefaultSettings {
        verbose: false,
        // 数値安定性のための収束許容誤差の緩和
        tol_gap_abs: 1e-5,
        tol_gap_rel: 1e-5,
        tol_feas: 1e-5,
        // 実行可能に近い問題が早期に非実行可能と
        // 宣言されないよう、非実行可能性検出を緩和
        tol_infeas_abs: 1e-5,
        tol_infeas_rel: 1e-5,
        // "AlmostSolved" の緩和許容誤差
        reduced_tol_gap_abs: 1e-3,
        reduced_tol_gap_rel: 1e-3,
        reduced_tol_feas: 1e-2,
        // より良いスケーリングのため均衡化の反復回数を増加
        equilibrate_max_iter: 50,
        // 制約行列の大きなスケール差に対応するため
        // 均衡化のスケーリング範囲を拡大
        equilibrate_min_scaling: 1e-6,
        equilibrate_max_scaling: 1e+6,
        // KKTシステムの安定化のため静的正則化を増加
        static_regularization_enable: true,
        static_regularization_constant: 1e-7,
        // 緩和された許容誤差で収束するために
        // より多くの反復が必要な場合に備え反復上限を増加
        max_iter: 500,
        ..DefaultSettings::default()
    }
}

// ─────────────────────────────────────────────
// アクティブインデックスの収集
// ─────────────────────────────────────────────

/// Z' ∪ Z'' からユニークなアクティブ点インデックスを収集する。
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
// 正則化エネルギー (Eq. 31, 33)
// ─────────────────────────────────────────────

/// 正則化の二次形式 P と線形項 q_reg を構築する。
///
/// (P_matrix, q_linear) を返す:
/// - P_matrix: n_vars × n_vars の二次形式（½x'Px 用）
/// - q_linear: 長さ n_vars のベクトル（q'x 用、目的関数に加算）
///
/// c¹, c² 部分（最初の 2*n_basis エントリ）のみ非ゼロエントリを持つ。
///
/// 正則化の種類:
/// - Biharmonic (Eq. 31): ∫∫ ||H_u||²_F + ||H_v||²_F dA（純二次）
/// - ARAP (Eq. 33): Σ_s (||J_A f(r_s)||² + ||J_S f(r_s) - d_s||²)
///   ARAPエネルギーは二次 + 線形 + 定数項に展開される。
///   線形項 (-2 d_s · J_S f) を q に含める必要がある。
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

    // Biharmonicエネルギー (Eq. 31)
    if lambda_bh > 0.0 {
        if let Some(ref bh) = precomputed.biharmonic_matrix {
            // bh は (2n × 2n)、λ * λ_bh * bh を P の c 部分に加算
            let scale = lambda * lambda_bh;
            for i in 0..2 * n_basis {
                for j in 0..2 * n_basis {
                    p[(i, j)] += scale * bh[(i, j)];
                }
            }
        }
    }

    // ARAPエネルギー (Eq. 33)
    // E_arap = Σ_s (||J_A f(r_s)||² + ||J_S f(r_s) - d_s||²)
    //
    // ||J_S - d||² を展開:
    //   = ||J_S||² - 2 d·J_S + ||d||²
    //   = ||J_S||² - 2(d_x·J_S_x + d_y·J_S_y) + 1
    //
    // よって E_arap = Σ_s (||J_A||² + ||J_S||² - 2 d_s·J_S + 1)
    //
    // 二次部分 → P行列: ||J_A||² + ||J_S||²
    // 線形部分 → qベクトル: -2 d_s·J_S
    // 定数部分（サンプルあたり1）: 無視（最適化に影響しない）
    //
    // 論文 Eq. 33: サンプル点 R'' は「正則化サンプル点の集合」
    // 安定な正則化のため、全ドメイン内部コロケーション点を使用する。
    // ドメイン輪郭外の点は除外される (Section 4)。
    if lambda_arap > 0.0 {
        let interior_indices: Vec<usize> = (0..state.collocation_points.len())
            .filter(|&i| state.domain_mask[i])
            .collect();
        let m_interior = interior_indices.len().max(1);
        let scale = lambda * lambda_arap / m_interior as f64; // 内部点数で正規化

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

            // 二次部分: ||J_A||² + ||J_S||²
            for i in 0..2 * n_basis {
                for j in 0..2 * n_basis {
                    p[(i, j)] += scale * (
                        a_ja_x[i] * a_ja_x[j] + a_ja_y[i] * a_ja_y[j]
                        + a_js_x[i] * a_js_x[j] + a_js_y[i] * a_js_y[j]
                    );
                }
            }

            // 線形部分: -2 d·J_S = -2(d_x·J_S_x + d_y·J_S_y)
            // J_S_x = a_js_x · c, J_S_y = a_js_y · c
            // → q への寄与: -2 * (d_x * a_js_x + d_y * a_js_y)
            for i in 0..2 * n_basis {
                q_reg[i] += scale * (-2.0) * (d.x * a_js_x[i] + d.y * a_js_y[i]);
            }
        }
    }

    (p, q_reg)
}

/// 数値積分によるbiharmonicエネルギー行列の構築 (Eq. 31)。
///
/// E_bh = ∫∫_Ω (||H_u||²_F + ||H_v||²_F) dA
///
/// H_u は u のヘシアン、||H||²_F = h_xx² + 2*h_xy² + h_yy²。
///
/// c の二次形式:
/// E_bh = c^T M_bh c
/// M_bh はドメイン上の数値積分で計算する。
pub fn build_biharmonic_matrix(
    basis: &dyn BasisFunction,
    collocation_points: &[Vector2<f64>],
    domain_bounds: &DomainBounds,
    n_basis: usize,
) -> DMatrix<f64> {
    let m = collocation_points.len();
    let dim = 2 * n_basis;
    let mut mat = DMatrix::zeros(dim, dim);

    // コロケーション点を等重み求積点として使用する。
    // 重み = ドメイン面積 / 点数（単純求積）
    let area =
        (domain_bounds.x_max - domain_bounds.x_min) * (domain_bounds.y_max - domain_bounds.y_min);
    let weight = area / m as f64;

    for pt in collocation_points {
        let (hxx, hxy, hyy) = basis.hessian(*pt);

        // u成分 (c¹): ||H_u||²_F = (Σ c¹_i h_xx_i)² + 2(Σ c¹_i h_xy_i)² + (Σ c¹_i h_yy_i)²
        // 二次形式: c¹^T (h_xx h_xx^T + 2 h_xy h_xy^T + h_yy h_yy^T) c¹
        // v成分 (c²) も右下ブロックで同様。

        for i in 0..n_basis {
            for j in 0..n_basis {
                let q = weight * (hxx[i] * hxx[j] + 2.0 * hxy[i] * hxy[j] + hyy[i] * hyy[j]);
                // uブロック（左上）
                mat[(i, j)] += q;
                // vブロック（右下）
                mat[(n_basis + i, n_basis + j)] += q;
            }
        }
    }

    mat
}

// ─────────────────────────────────────────────
// ユーティリティ: clarabel用の疎行列構築
// ─────────────────────────────────────────────

/// 疎行表現をclarabel用のCscMatrixに変換する。
fn sparse_rows_to_csc(
    rows: &[Vec<(usize, f64)>],
    b: &[f64],
    n_cols: usize,
) -> (CscMatrix<f64>, Vec<f64>) {
    let n_rows = rows.len();

    // 列ごとのエントリ数をカウント
    let mut col_counts = vec![0usize; n_cols];
    for row in rows {
        for &(col, _) in row {
            col_counts[col] += 1;
        }
    }

    // colptr を構築
    let mut colptr = vec![0usize; n_cols + 1];
    for j in 0..n_cols {
        colptr[j + 1] = colptr[j] + col_counts[j];
    }
    let nnz = colptr[n_cols];

    // rowval と nzval を充填
    let mut rowval = vec![0usize; nnz];
    let mut nzval = vec![0.0f64; nnz];
    let mut col_pos = vec![0usize; n_cols]; // 列ごとの現在の挿入位置

    for (row_idx, row) in rows.iter().enumerate() {
        for &(col, val) in row {
            let pos = colptr[col] + col_pos[col];
            rowval[pos] = row_idx;
            nzval[pos] = val;
            col_pos[col] += 1;
        }
    }

    // 各列内のエントリを行インデックスでソート（CscMatrixの要件）
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

/// 密な対称行列を上三角CscMatrixに変換する。
/// Clarabel は P が上三角であることを要求する。
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

/// ソルバーを破壊するNaN/Inf値がないか問題データを検査する。
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
