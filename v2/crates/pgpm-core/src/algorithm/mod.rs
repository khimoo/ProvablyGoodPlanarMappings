//! Algorithm 1 本体 + 周辺ロジック。
//!
//! 論文参照: Algorithm 1 (Section 5)
//! このモジュールは証明付き良好な平面写像の `Algorithm` 構造体を提供する。
//! `PgpmAlgorithm` トレイトと `MappingBridge` トレイトは廃止され、
//! 全てのロジックがこの構造体の `impl` に直接配置されている。

pub mod active_set;
pub mod strategy;

use crate::basis::BasisFunction;
use crate::distortion;
use crate::model::domain::Domain;
use crate::model::types::*;
use crate::numerics::solver;
use crate::policy::DistortionPolicy;
use log::warn;
use nalgebra::{DMatrix, Vector2};

/// Algorithm 1: 歪みポリシーを動的ディスパッチで保持する完全実装。
pub struct Algorithm {
    /// 基底関数 (Table 1)
    basis: Box<dyn BasisFunction>,
    /// アルゴリズムパラメータ (K, lambda, 正則化タイプ)
    params: MappingParams,
    /// 歪みポリシー (isometric または conformal)
    policy: Box<dyn DistortionPolicy>,
    /// アルゴリズム状態 (係数、アクティブ集合、フレーム等)
    state: AlgorithmState,
    /// ソースハンドル位置 {p_l} (Eq. 29) -- 固定
    source_handles: Vec<Vector2<f64>>,
    /// biharmonic エネルギー積分用ドメイン境界
    domain_bounds: DomainBounds,
    /// "x ∈ Ω" テスト用ドメイン Ω (論文 Section 4)。
    /// `None` の場合、全グリッド点がドメイン内とみなされる。
    domain: Option<Box<dyn Domain>>,
    /// SOCP ソルバーの数値調整（論文に無い）。
    solver_config: SolverConfig,
}

impl Algorithm {
    /// 新しい Algorithm インスタンスを作成。
    ///
    /// # 引数
    /// - `basis`: 基底関数実装 (Gaussian, B-Spline, TPS)
    /// - `params`: アルゴリズムパラメータ (K, lambda, 正則化)
    /// - `policy`: 歪みポリシー (isometric または conformal)
    /// - `domain_bounds`: ドメイン Ω のバウンディングボックス (Eq. 5)
    /// - `source_handles`: 固定ハンドル位置 {p_l} (Eq. 29)
    /// - `grid_resolution`: 1辺あたりのグリッド点数 (論文 Section 6: 200)
    /// - `fps_k`: Z''（安定集合）の最遠点サンプル数
    /// - `domain`: 抽象ドメイン Ω。`Some` の場合、`domain.contains(pt)` が
    ///   true を返すグリッド点のみがアクティブ/安定集合と ARAP 正則化の
    ///   対象となる。矩形グリッド構造は局所最大検出用に保持される
    ///   (Section 5)。`None` の場合、全グリッド点がドメイン内とみなされる。
    pub fn new(
        basis: Box<dyn BasisFunction>,
        params: MappingParams,
        policy: Box<dyn DistortionPolicy>,
        domain_bounds: DomainBounds,
        source_handles: Vec<Vector2<f64>>,
        grid_resolution: usize,
        fps_k: usize,
        domain: Option<Box<dyn Domain>>,
        solver_config: SolverConfig,
    ) -> Self {
        // コロケーショングリッドを生成
        // Section 4: 「ドメイン内に収まる周囲一様グリッドの
        // 全点を考慮する」
        let (collocation_points, grid_width, grid_height) =
            generate_collocation_grid(&domain_bounds, grid_resolution);

        let m = collocation_points.len();

        // ドメインマスクを構築: Ω 内の点に対して true。
        // 論文 Section 4: 「ドメイン内に収まる周囲一様グリッドの
        // 全点を考慮する」
        let domain_mask = build_domain_mask(&collocation_points, domain.as_deref());

        // Section 5 "Activation of constraints":
        // K_high = 0.1 + 0.9*K、K_low = 0.5 + 0.5*K
        let k = params.k_bound;
        let k_high = 0.1 + 0.9 * k;
        let k_low = 0.5 + 0.5 * k;

        // フレームを (1, 0) で初期化 -- Algorithm 1: "Initialize d_i"
        let frames = vec![Vector2::new(1.0, 0.0); m];

        // 恒等係数で初期化
        let coefficients = basis.identity_coefficients();

        // FPS で安定集合を初期化（ドメイン内部の点のみ）
        let stable_set =
            active_set::initialize_stable_set(&collocation_points, fps_k, &domain_mask);

        let state = AlgorithmState {
            coefficients,
            collocation_points,
            active_set: Vec::new(), // Algorithm 1: "空のアクティブ集合 Z' を初期化"
            stable_set,
            frames,
            k_high,
            k_low,
            domain_mask,
            precomputed: None,
            grid_width,
            grid_height,
            prev_target_handles: None,
        };

        Self {
            basis,
            params,
            policy,
            state,
            source_handles,
            domain_bounds,
            domain,
            solver_config,
        }
    }

    // ─────────────────────────────────────────
    // 公開アクセサ
    // ─────────────────────────────────────────

    /// 現在の係数行列 c を取得（Eq. 3）。
    pub fn coefficients(&self) -> &CoefficientMatrix {
        &self.state.coefficients
    }

    /// 基底関数の参照を取得（Table 1）。
    pub fn basis(&self) -> &dyn BasisFunction {
        self.basis.as_ref()
    }

    /// 現在のアルゴリズムパラメータを取得。
    pub fn params(&self) -> &MappingParams {
        &self.params
    }

    /// 外部検査用にアルゴリズム状態（不変）を取得。
    pub fn state(&self) -> &AlgorithmState {
        &self.state
    }

    /// 歪みポリシーの参照を取得。
    pub fn policy(&self) -> &dyn DistortionPolicy {
        self.policy.as_ref()
    }

    /// 現在のアクティブ集合サイズを取得。
    pub fn active_set_size(&self) -> usize {
        self.state.active_set.len()
    }

    /// コロケーション点を取得。
    pub fn collocation_points(&self) -> &[Vector2<f64>] {
        &self.state.collocation_points
    }

    /// Strategy 2 精緻化の最大グリッド解像度。
    pub fn max_refinement_resolution(&self) -> usize {
        self.solver_config.max_refinement_resolution
    }

    /// コロケーショングリッドの解像度（幅、高さ）（Section 4）。
    pub fn grid_resolution(&self) -> (usize, usize) {
        (self.state.grid_width, self.state.grid_height)
    }

    /// コロケーション点の数 |Z|（Section 4）。
    pub fn num_collocation_points(&self) -> usize {
        self.state.collocation_points.len()
    }

    /// 基底関数の数 n（Table 1）。
    pub fn num_basis_functions(&self) -> usize {
        self.basis.count()
    }

    /// ソースハンドル位置 {p_l}（Eq. 29）。
    pub fn source_handles(&self) -> &[Vector2<f64>] {
        &self.source_handles
    }

    /// ドメイン Ω のバウンディングボックス（Eq. 5）。
    pub fn domain_bounds(&self) -> &DomainBounds {
        &self.domain_bounds
    }

    // ─────────────────────────────────────────
    // Algorithm 1 本体
    // ─────────────────────────────────────────

    /// 全コロケーション点で基底関数の値と勾配を事前計算。
    /// Algorithm 1: "if first step then" ブロック。
    ///
    /// `state.precomputed` が既に `Some` の場合は何もしない。
    pub fn ensure_precomputed(&mut self) {
        if self.state.precomputed.is_some() {
            return;
        }

        let (phi, grad_phi_x, grad_phi_y) = self.compute_basis_matrices();

        self.state.precomputed = Some(PrecomputedData {
            phi,
            grad_phi_x,
            grad_phi_y,
            biharmonic_matrix: None,
        });

        // 現在の正則化タイプが必要な場合、biharmonic行列を構築
        let needs_bh = matches!(
            self.params.regularization,
            RegularizationType::Biharmonic | RegularizationType::Mixed { .. }
        );
        if needs_bh {
            ensure_biharmonic_matrix_inner(
                self.basis.as_ref(),
                &self.domain_bounds,
                &mut self.state,
            );
        }
    }

    /// 全コロケーション点で基底関数の値と勾配を評価。
    ///
    /// (phi, grad_phi_x, grad_phi_y) 行列を返す。
    /// 各行列の形状は (num_collocation, num_basis)。
    fn compute_basis_matrices(&self) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let m = self.state.collocation_points.len();
        let n = self.basis.count();

        let mut phi = DMatrix::zeros(m, n);
        let mut grad_phi_x = DMatrix::zeros(m, n);
        let mut grad_phi_y = DMatrix::zeros(m, n);

        let mut nan_inf_count = 0usize;
        for (idx, pt) in self.state.collocation_points.iter().enumerate() {
            let val = self.basis.evaluate(*pt);
            let (gx, gy) = self.basis.gradient(*pt);

            for i in 0..n {
                // 基底関数からの NaN/Inf を防護（形状認識基底で
                // ドメイン境界付近の測地距離が無限大になる場合に発生しうる）。
                if !val[i].is_finite() || !gx[i].is_finite() || !gy[i].is_finite() {
                    nan_inf_count += 1;
                }
                phi[(idx, i)] = if val[i].is_finite() { val[i] } else { 0.0 };
                grad_phi_x[(idx, i)] = if gx[i].is_finite() { gx[i] } else { 0.0 };
                grad_phi_y[(idx, i)] = if gy[i].is_finite() { gy[i] } else { 0.0 };
            }
        }
        if nan_inf_count > 0 {
            warn!(
                "Precompute: {} NaN/Inf basis values replaced with 0.0 \
                 (expected near domain boundaries with shape-aware bases)",
                nan_inf_count,
            );
        }

        (phi, grad_phi_x, grad_phi_y)
    }

    /// Algorithm 1: 1ステップを実行（Section 5）。
    ///
    /// 擬似コードとの対応:
    /// 1. [初回ステップのみ] phi(z), grad_phi(z) を事前計算
    /// 2. 全 z ∈ Z で D(z) を評価
    /// 3. Z_max（D の局所最大）を見つける
    /// 4. D(z) > K_high となる z ∈ Z_max を Z' に追加
    /// 5. D(z) < K_low となる z ∈ Z' を Z' から削除
    /// 6. SOCP を求解（Eq. 18）→ c を更新
    /// 7. d_i を更新（Eq. 27）
    pub fn step(&mut self, target_handles: &[Vector2<f64>]) -> Result<StepInfo, AlgorithmError> {
        self.ensure_precomputed();

        // ハンドル数の整合性を検証
        {
            let n_src = self.source_handles.len();
            let n_tgt = target_handles.len();
            if n_src != n_tgt {
                return Err(AlgorithmError::InvalidInput(format!(
                    "source_handles ({}) and target_handles ({}) count mismatch",
                    n_src, n_tgt,
                )));
            }
        }

        // 2. 歪みを評価（Eq. 19-20）
        let distortions = {
            let precomputed = self.state.precomputed.as_ref().ok_or_else(|| {
                AlgorithmError::InvalidInput("Precomputed data not available (bug)".into())
            })?;
            distortion::evaluate_distortion_all(
                &self.state.coefficients,
                precomputed,
                self.policy.as_ref(),
            )
        };

        // 3-5. アクティブ集合を更新
        let prev_active_set = self.state.active_set.clone();
        active_set::update_active_set(&mut self.state, &distortions);

        let max_distortion = distortions
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // 収束を確認
        let n_active = self.state.active_set.len();
        let changed = self.state.prev_target_handles.as_deref() != Some(target_handles);
        self.state.prev_target_handles = Some(target_handles.to_vec());

        let converged = !changed
            && max_distortion <= self.params.k_bound
            && self.state.active_set == prev_active_set;

        // 6. SOCP を求解（Eq. 18）
        let new_coefficients = {
            self.state.precomputed.as_ref().ok_or_else(|| {
                AlgorithmError::InvalidInput("Precomputed data not available (bug)".into())
            })?;
            solver::solve_socp(
                &self.source_handles,
                target_handles,
                self.basis.as_ref(),
                &self.state,
                &self.params,
                self.policy.as_ref(),
                &self.solver_config,
            )?
        };

        // 7. 係数とフレームを更新（Eq. 27）
        self.state.coefficients = new_coefficients;

        // フレーム更新のため全点で J_S を評価（Eq. 27）
        let j_s_values = {
            let pre = self.state.precomputed.as_ref().ok_or_else(|| {
                AlgorithmError::InvalidInput("Precomputed data not available (bug)".into())
            })?;
            distortion::evaluate_j_s_all(&self.state.coefficients, pre)
        };

        let eps = 1e-10;
        let indices: Vec<usize> = self
            .state
            .active_set
            .iter()
            .chain(self.state.stable_set.iter())
            .copied()
            .collect();
        for idx in indices {
            let j_s = j_s_values[idx];
            let norm = j_s.norm();
            if norm > eps {
                self.state.frames[idx] = j_s / norm;
            }
        }
        let stable_set_size = self.state.stable_set.len();

        Ok(StepInfo {
            max_distortion,
            active_set_size: n_active,
            stable_set_size,
            converged,
        })
    }

    /// 実行時にアルゴリズムパラメータを更新（K, lambda, 正則化）。
    pub fn update_params(&mut self, params: MappingParams) {
        let k = params.k_bound;
        self.state.k_high = 0.1 + 0.9 * k;
        self.state.k_low = 0.5 + 0.5 * k;

        let needs_bh = matches!(
            params.regularization,
            RegularizationType::Biharmonic | RegularizationType::Mixed { .. }
        );
        let has_bh = self
            .state
            .precomputed
            .as_ref()
            .map_or(false, |p| p.biharmonic_matrix.is_some());

        if needs_bh && !has_bh {
            ensure_biharmonic_matrix_inner(
                self.basis.as_ref(),
                &self.domain_bounds,
                &mut self.state,
            );
        }

        self.params = params;
    }

    /// Strategy 2 事後精緻化（Section 5 "Strategies"）。
    ///
    /// Algorithm 側の委譲メソッド。実体は `strategy::refine_to_target`。
    pub fn refine_strategy2(
        &mut self,
        k_max: f64,
        target_handles: &[Vector2<f64>],
    ) -> Result<strategy::Strategy2Result, AlgorithmError> {
        strategy::refine_to_target(self, k_max, target_handles)
    }

    /// 複数点で順方向写像 f(x) = Σ c_i φ_i(x) を評価（Eq. 3）。
    ///
    /// ドメイン空間の点のスライスを受け取り、写像後の位置を返す。
    /// 任意の基底関数タイプに対する CPU レンダリングパスで使用。
    pub fn evaluate_mapping_at(&self, points: &[Vector2<f64>]) -> Vec<Vector2<f64>> {
        let c = &self.state.coefficients;
        let n = self.basis.count();
        points
            .iter()
            .map(|&x| {
                let phi = self.basis.evaluate(x);
                let mut u = 0.0;
                let mut v = 0.0;
                for i in 0..n {
                    u += c[(0, i)] * phi[i];
                    v += c[(1, i)] * phi[i];
                }
                Vector2::new(u, v)
            })
            .collect()
    }

    /// 点 x でヤコビアン勾配（∇u, ∇v）を計算（Eq. 3 を微分）。
    pub fn grad_uv_at(&self, x: Vector2<f64>) -> (Vector2<f64>, Vector2<f64>) {
        let (gx, gy) = self.basis.gradient(x);
        let c = &self.state.coefficients;
        let n = self.basis.count();

        let mut grad_u = Vector2::new(0.0, 0.0);
        let mut grad_v = Vector2::new(0.0, 0.0);
        for i in 0..n {
            grad_u.x += c[(0, i)] * gx[i];
            grad_u.y += c[(0, i)] * gy[i];
            grad_v.x += c[(1, i)] * gx[i];
            grad_v.y += c[(1, i)] * gy[i];
        }
        (grad_u, grad_v)
    }

    /// 点 x で J_S f(x) と J_A f(x) を計算（Eq. 19-20）。
    pub fn j_s_j_a_at(&self, x: Vector2<f64>) -> (Vector2<f64>, Vector2<f64>) {
        let (grad_u, grad_v) = self.grad_uv_at(x);
        distortion::compute_j_s_j_a(grad_u, grad_v)
    }

    /// 点 x で特異値（Σ, σ）を計算（Eq. 20）。
    pub fn singular_values_at(&self, x: Vector2<f64>) -> (f64, f64) {
        let (j_s, j_a) = self.j_s_j_a_at(x);
        distortion::singular_values(j_s, j_a)
    }

    /// コロケーショングリッドを新しい解像度で再構築（Strategy 2 用）。
    /// 現在の係数を保持。アクティブ集合、フレーム、事前計算データをリセット。
    pub fn rebuild_grid(&mut self, new_resolution: usize) {
        let (new_points, new_gw, new_gh) =
            generate_collocation_grid(&self.domain_bounds, new_resolution);
        let m = new_points.len();
        let new_mask = build_domain_mask(&new_points, self.domain.as_deref());
        let new_frames = vec![Vector2::new(1.0, 0.0); m];

        let fps_k = self.state.stable_set.len().max(4);
        let new_stable_set =
            active_set::initialize_stable_set(&new_points, fps_k, &new_mask);

        let coefficients = self.state.coefficients.clone();

        self.state = AlgorithmState {
            coefficients,
            collocation_points: new_points,
            active_set: Vec::new(),
            stable_set: new_stable_set,
            frames: new_frames,
            k_high: self.state.k_high,
            k_low: self.state.k_low,
            domain_mask: new_mask,
            precomputed: None,
            grid_width: new_gw,
            grid_height: new_gh,
            prev_target_handles: None,
        };
    }
}

// ─────────────────────────────────────────────
// プライベートヘルパー
// ─────────────────────────────────────────────

/// 事前計算データに biharmonic 行列（Eq. 31）を構築。
fn ensure_biharmonic_matrix_inner(
    basis: &dyn BasisFunction,
    domain_bounds: &DomainBounds,
    state: &mut AlgorithmState,
) {
    if let Some(ref mut precomputed) = state.precomputed {
        if precomputed.biharmonic_matrix.is_none() {
            let n = basis.count();
            precomputed.biharmonic_matrix = Some(solver::build_biharmonic_matrix(
                basis,
                &state.collocation_points,
                domain_bounds,
                n,
            ));
        }
    }
}

/// ドメイン境界内に一様コロケーショングリッドを生成。
///
/// Section 4: 「ドメイン内に収まる周囲一様グリッドの全点を考慮する」
pub(crate) fn generate_collocation_grid(
    bounds: &DomainBounds,
    resolution: usize,
) -> (Vec<Vector2<f64>>, usize, usize) {
    let dx = (bounds.x_max - bounds.x_min) / (resolution as f64 - 1.0);
    let dy = (bounds.y_max - bounds.y_min) / (resolution as f64 - 1.0);

    let mut points = Vec::with_capacity(resolution * resolution);

    let res_x = resolution;
    let res_y = resolution;

    for row in 0..res_y {
        for col in 0..res_x {
            let x = bounds.x_min + col as f64 * dx;
            let y = bounds.y_min + row as f64 * dy;
            points.push(Vector2::new(x, y));
        }
    }

    (points, res_x, res_y)
}

/// コロケーション点とオプションのドメインからドメインマスクを構築。
///
/// 論文 Section 4: 「ドメイン内に収まる周囲一様グリッドの全点を考慮する」
pub(crate) fn build_domain_mask(
    points: &[Vector2<f64>],
    domain: Option<&dyn Domain>,
) -> Vec<bool> {
    match domain {
        Some(d) => points.iter().map(|pt| d.contains(pt)).collect(),
        None => vec![true; points.len()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::gaussian::GaussianBasis;
    use crate::policy::IsometricPolicy;

    fn make_test_algorithm() -> Algorithm {
        // [0,1]² 上の単純な4中心 Gaussian 基底
        let centers = vec![
            Vector2::new(0.25, 0.25),
            Vector2::new(0.75, 0.25),
            Vector2::new(0.25, 0.75),
            Vector2::new(0.75, 0.75),
        ];
        let basis = Box::new(GaussianBasis::new(centers, 0.3));

        let params = MappingParams {
            k_bound: 3.0,
            lambda_reg: 0.0, // 単純化のため正則化なし
            regularization: RegularizationType::Arap,
        };

        let domain = DomainBounds {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        };

        let handles = vec![Vector2::new(0.5, 0.5)];

        // テスト用の小さいグリッド（ドメイン制約なし → 完全矩形）
        Algorithm::new(
            basis,
            params,
            Box::new(IsometricPolicy),
            domain,
            handles,
            10,
            4,
            None,
            SolverConfig::default(),
        )
    }

    #[test]
    fn test_identity_mapping() {
        let alg = make_test_algorithm();

        // 恒等係数では f(x) は近似的に x であるべき (Eq. 3)
        let test_point = Vector2::new(0.5, 0.5);
        let phi = alg.basis().evaluate(test_point);
        let c = alg.coefficients();
        let n = alg.basis().count();
        let mut u = 0.0;
        let mut v = 0.0;
        for i in 0..n {
            u += c[(0, i)] * phi[i];
            v += c[(1, i)] * phi[i];
        }
        let result = Vector2::new(u, v);

        assert!(
            (result - test_point).norm() < 1e-10,
            "Identity mapping: expected ({}, {}), got ({}, {})",
            test_point.x,
            test_point.y,
            result.x,
            result.y
        );
    }

    #[test]
    fn test_identity_distortion_is_one() {
        use crate::distortion;

        let mut alg = make_test_algorithm();
        alg.ensure_precomputed();

        let precomputed = alg.state.precomputed.as_ref().unwrap();
        let distortions = distortion::evaluate_distortion_all(
            &alg.state.coefficients,
            precomputed,
            alg.policy.as_ref(),
        );

        // 恒等写像は全点で D ≈ 1 であるべき
        for (i, &d) in distortions.iter().enumerate() {
            assert!(
                (d - 1.0).abs() < 1e-6,
                "Distortion at point {} = {}, expected ~1.0",
                i,
                d
            );
        }
    }

    #[test]
    fn test_step_with_identity_target() {
        let mut alg = make_test_algorithm();

        // ターゲット = ソース（恒等変形）
        let target = vec![Vector2::new(0.5, 0.5)];
        let info = alg.step(&target).expect("Step should succeed");

        // 恒等ターゲットで求解後、歪みは 1 に近いはず
        assert!(
            info.max_distortion < 2.0,
            "Max distortion = {}, expected < 2.0 for near-identity",
            info.max_distortion
        );
    }

    #[test]
    fn test_step_with_deformed_target() {
        let mut alg = make_test_algorithm();

        // ハンドルを少し動かす
        let target = vec![Vector2::new(0.6, 0.5)];
        let info = alg.step(&target).expect("Step should succeed");

        // 写像はまだ有効であるべき（無限大の歪みではない）
        assert!(
            info.max_distortion < 100.0,
            "Max distortion = {}, expected finite value",
            info.max_distortion
        );

        // ハンドルでの評価点はターゲットに近いはず (Eq. 3)
        let phi = alg.basis().evaluate(Vector2::new(0.5, 0.5));
        let c = alg.coefficients();
        let n = alg.basis().count();
        let mut mu = 0.0;
        let mut mv = 0.0;
        for i in 0..n {
            mu += c[(0, i)] * phi[i];
            mv += c[(1, i)] * phi[i];
        }
        let mapped = Vector2::new(mu, mv);
        assert!(
            (mapped - Vector2::new(0.6, 0.5)).norm() < 0.5,
            "Handle should approximately reach target: got ({}, {})",
            mapped.x,
            mapped.y
        );
    }

    #[test]
    fn test_collocation_grid() {
        let bounds = DomainBounds {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        };
        let (points, w, h) = generate_collocation_grid(&bounds, 5);
        assert_eq!(w, 5);
        assert_eq!(h, 5);
        assert_eq!(points.len(), 25);

        // コーナーを確認
        assert!((points[0] - Vector2::new(0.0, 0.0)).norm() < 1e-12);
        assert!((points[4] - Vector2::new(1.0, 0.0)).norm() < 1e-12);
        assert!((points[24] - Vector2::new(1.0, 1.0)).norm() < 1e-12);
    }

    #[test]
    fn test_multiple_steps() {
        let mut alg = make_test_algorithm();
        let target = vec![Vector2::new(0.6, 0.5)];

        // 複数ステップ実行 -- 歪みは収束するはず
        for step in 0..5 {
            let info = alg.step(&target).expect("Step should succeed");
            println!(
                "Step {}: max_dist={:.4}, active={}",
                step, info.max_distortion, info.active_set_size
            );
            // 最初のステップ後、歪みは有限であるべき
            assert!(info.max_distortion.is_finite());
        }
    }

    #[test]
    fn test_step_handle_count_mismatch() {
        let mut alg = make_test_algorithm();

        // ターゲット数 > ソース数 → エラー
        let too_many = vec![Vector2::new(0.5, 0.5), Vector2::new(0.6, 0.6)];
        let err = alg.step(&too_many).unwrap_err();
        assert!(
            matches!(err, AlgorithmError::InvalidInput(_)),
            "Expected InvalidInput, got {:?}",
            err
        );

        // ターゲット数 = 0 → エラー
        let empty: Vec<Vector2<f64>> = vec![];
        let err = alg.step(&empty).unwrap_err();
        assert!(
            matches!(err, AlgorithmError::InvalidInput(_)),
            "Expected InvalidInput for empty targets, got {:?}",
            err
        );
    }

    #[test]
    fn test_domain_mask_with_polygon_domain() {
        use crate::model::domain::PolygonDomain;

        let centers = vec![Vector2::new(0.5, 0.5)];
        let basis = Box::new(GaussianBasis::new(centers.clone(), 0.3));
        let params = MappingParams {
            k_bound: 3.0,
            lambda_reg: 0.0,
            regularization: RegularizationType::Arap,
        };
        let domain = DomainBounds {
            x_min: 0.0,
            x_max: 1.0,
            y_min: 0.0,
            y_max: 1.0,
        };

        // 単位正方形のコーナーを除外するひし形輪郭
        let contour = vec![
            Vector2::new(0.5, 0.0),
            Vector2::new(1.0, 0.5),
            Vector2::new(0.5, 1.0),
            Vector2::new(0.0, 0.5),
        ];

        let polygon_domain = PolygonDomain::new(contour, vec![]);
        let alg = Algorithm::new(
            basis,
            params,
            Box::new(IsometricPolicy),
            domain,
            centers,
            5,
            2,
            Some(Box::new(polygon_domain)),
            SolverConfig::default(),
        );

        // グリッドは 5x5 = 25 点だが、一部はマスクされるべき
        let state = alg.state();
        let mask = &state.domain_mask;
        assert_eq!(mask.len(), 25);

        // グリッドのコーナー (0,0), (1,0), (0,1), (1,1) は外側であるべき
        assert!(!mask[0], "(0,0) should be outside diamond");
        assert!(!mask[4], "(1,0) should be outside diamond");
        assert!(!mask[20], "(0,1) should be outside diamond");
        assert!(!mask[24], "(1,1) should be outside diamond");

        // 中心 (0.5, 0.5) は内側であるべき
        assert!(mask[12], "(0.5,0.5) should be inside diamond");

        // 安定集合はドメイン内部の点のみを含むべき
        for &idx in &state.stable_set {
            assert!(mask[idx], "Stable set point {} should be inside domain", idx);
        }
    }
}
