//! 平面写像トレイト。
//!
//! [`PlanarMapping`] はアルゴリズム全体（Algorithm 1, Section 5）の抽象定義。
//! 数学的評価（Eq. 3, Section 3）と最適化手続き（SOCP、アクティブ集合、
//! Strategy 2）の両方を含む。
//!
//! **設計**: `parts()`/`parts_mut()` パターンで不変コンテキスト
//! ([`MappingContext`]) と可変状態 ([`AlgorithmState`]) の借用を分離。
//! これにより Algorithm 1 のロジック全体をデフォルトメソッドに配置し、
//! 具象型は借用分離アクセサと `set_params` のみを提供すれば良い。

use crate::algorithm::active_set;
use crate::algorithm::strategy;
use crate::basis::BasisFunction;
use crate::distortion;
use crate::model::types::{
    AlgorithmError, AlgorithmState, CoefficientMatrix, DomainBounds, MappingContext,
    MappingParams, PrecomputedData, RegularizationType, StepInfo,
};
use crate::numerics::solver;
use log::warn;
use nalgebra::{DMatrix, Vector2};

/// 証明付き良好な平面写像の抽象定義（Algorithm 1, Section 5）。
///
/// このトレイトはアルゴリズム全体を捉える：初期化、SOCP最適化、
/// アクティブ集合管理、フレーム更新、Strategy 2 精緻化。
///
/// **必須メソッド**（3つのみ）:
/// - [`parts`](PlanarMapping::parts) / [`parts_mut`](PlanarMapping::parts_mut) —
///   `(MappingContext, &AlgorithmState)` または `(MappingContext, &mut AlgorithmState)`
///   を返す借用分離アクセサ。
/// - [`set_params`](PlanarMapping::set_params) — アルゴリズムパラメータの更新。
///
/// **デフォルトメソッド**: Algorithm 1 の完全なスケルトンと
/// 数学的評価（Eq. 3）。
pub trait PlanarMapping: Send + Sync {
    // ─────────────────────────────────────────
    // 必須: 借用分離アクセサ
    // ─────────────────────────────────────────

    /// self を不変コンテキストと不変状態に分離。
    fn parts(&self) -> (MappingContext<'_>, &AlgorithmState);

    /// self を不変コンテキストと可変状態に分離。
    fn parts_mut(&mut self) -> (MappingContext<'_>, &mut AlgorithmState);

    /// 新しいアルゴリズムパラメータ（K, lambda, 正則化）を格納。
    /// AlgorithmState の外部にある params フィールドを変更する。
    fn set_params(&mut self, params: MappingParams);

    // ─────────────────────────────────────────
    // デフォルトメソッド: Algorithm 1 スケルトン
    // ─────────────────────────────────────────

    /// 現在の係数行列 c を取得（Eq. 3）。
    fn coefficients(&self) -> &CoefficientMatrix {
        let (_, state) = self.parts();
        &state.coefficients
    }

    /// 基底関数の参照を取得（Table 1）。
    fn basis(&self) -> &dyn BasisFunction {
        let (ctx, _) = self.parts();
        ctx.basis
    }

    /// 現在のアルゴリズムパラメータを取得。
    fn params(&self) -> MappingParams {
        let (ctx, _) = self.parts();
        ctx.params.clone()
    }

    /// 外部検査用にアルゴリズム状態（不変）を取得。
    fn state(&self) -> &AlgorithmState {
        let (_, state) = self.parts();
        state
    }

    /// Strategy 2 精緻化の最大グリッド解像度。
    fn max_refinement_resolution(&self) -> usize {
        let (ctx, _) = self.parts();
        ctx.solver_config.max_refinement_resolution
    }

    /// コロケーショングリッドの解像度（幅、高さ）（Section 4）。
    fn grid_resolution(&self) -> (usize, usize) {
        let (_, state) = self.parts();
        (state.grid_width, state.grid_height)
    }

    /// コロケーション点の数 |Z|（Section 4）。
    fn num_collocation_points(&self) -> usize {
        let (_, state) = self.parts();
        state.collocation_points.len()
    }

    /// 基底関数の数 n（Table 1）。
    fn num_basis_functions(&self) -> usize {
        let (ctx, _) = self.parts();
        ctx.basis.count()
    }

    /// ソースハンドル位置 {p_l}（Eq. 29）。
    fn source_handles(&self) -> Vec<Vector2<f64>> {
        let (ctx, _) = self.parts();
        ctx.source_handles.to_vec()
    }

    /// ドメイン Ω のバウンディングボックス（Eq. 5）。
    fn domain_bounds_query(&self) -> DomainBounds {
        let (ctx, _) = self.parts();
        ctx.domain_bounds.clone()
    }

    /// 全コロケーション点で基底関数の値と勾配を事前計算。
    /// Algorithm 1: "if first step then" ブロック。
    ///
    /// `state.precomputed` が既に `Some` の場合は何もしない。
    ///
    /// 手続き分解:
    /// 1. [`Self::compute_basis_matrices`] — φ_i(z), ∇φ_i(z) を評価
    /// 2. [`PrecomputedData`] として格納
    /// 3. 必要なら Eq. 31 biharmonic 行列を構築
    fn ensure_precomputed(&mut self) {
        {
            let (_, state) = self.parts();
            if state.precomputed.is_some() {
                return;
            }
        }

        let (phi, grad_phi_x, grad_phi_y) = self.compute_basis_matrices();

        let (ctx, state) = self.parts_mut();
        state.precomputed = Some(PrecomputedData {
            phi,
            grad_phi_x,
            grad_phi_y,
            biharmonic_matrix: None,
        });

        // 現在の正則化タイプが必要な場合、biharmonic行列を構築
        let needs_bh = matches!(
            ctx.params.regularization,
            RegularizationType::Biharmonic | RegularizationType::Mixed { .. }
        );
        if needs_bh {
            ensure_biharmonic_matrix_inner(ctx.basis, ctx.domain_bounds, state);
        }
    }

    /// 全コロケーション点で基底関数の値と勾配を評価。
    ///
    /// (phi, grad_phi_x, grad_phi_y) 行列を返す。
    /// 各行列の形状は (num_collocation, num_basis)。
    ///
    /// NaN/Inf 値（ドメイン境界付近の形状認識基底から発生）は
    /// 0.0 に置換され、警告がログに記録される。
    fn compute_basis_matrices(&self) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let (ctx, state) = self.parts();
        let m = state.collocation_points.len();
        let n = ctx.basis.count();

        let mut phi = DMatrix::zeros(m, n);
        let mut grad_phi_x = DMatrix::zeros(m, n);
        let mut grad_phi_y = DMatrix::zeros(m, n);

        let mut nan_inf_count = 0usize;
        for (idx, pt) in state.collocation_points.iter().enumerate() {
            let val = ctx.basis.evaluate(*pt);
            let (gx, gy) = ctx.basis.gradient(*pt);

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
    fn step(&mut self, target_handles: &[Vector2<f64>]) -> Result<StepInfo, AlgorithmError> {
        self.ensure_precomputed();

        // ハンドル数の整合性を検証
        {
            let (ctx, _) = self.parts();
            let n_src = ctx.source_handles.len();
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
            let (ctx, state) = self.parts();
            let precomputed = state.precomputed.as_ref().ok_or_else(|| {
                AlgorithmError::InvalidInput("Precomputed data not available (bug)".into())
            })?;
            distortion::evaluate_distortion_all(
                &state.coefficients,
                precomputed,
                ctx.policy,
            )
        };

        // 3-5. アクティブ集合を更新
        let prev_active_set = {
            let (_, state) = self.parts();
            state.active_set.clone()
        };
        {
            let (_, state) = self.parts_mut();
            active_set::update_active_set(state, &distortions);
        }

        let max_distortion = distortions
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // 収束を確認
        let (n_active, converged) = {
            let (ctx, state) = self.parts_mut();
            let n_active = state.active_set.len();

            let changed = state.prev_target_handles.as_deref() != Some(target_handles);
            state.prev_target_handles = Some(target_handles.to_vec());

            let converged = !changed
                && max_distortion <= ctx.params.k_bound
                && state.active_set == prev_active_set;

            (n_active, converged)
        };

        // 6. SOCP を求解（Eq. 18）
        let new_coefficients = {
            let (ctx, state) = self.parts();
            state.precomputed.as_ref().ok_or_else(|| {
                AlgorithmError::InvalidInput("Precomputed data not available (bug)".into())
            })?;
            solver::solve_socp(
                ctx.source_handles,
                target_handles,
                ctx.basis,
                state,
                ctx.params,
                ctx.policy,
                ctx.solver_config,
            )?
        };

        // 7. 係数とフレームを更新（Eq. 27）
        {
            let (_, state) = self.parts_mut();
            state.coefficients = new_coefficients;
        }

        // フレーム更新のため全点で J_S を評価（Eq. 27）
        let j_s_values = {
            let (_, state) = self.parts();
            let pre = state.precomputed.as_ref().ok_or_else(|| {
                AlgorithmError::InvalidInput("Precomputed data not available (bug)".into())
            })?;
            distortion::evaluate_j_s_all(&state.coefficients, pre)
        };

        let stable_set_size = {
            let (_, state) = self.parts_mut();
            let eps = 1e-10;
            let indices: Vec<usize> = state
                .active_set
                .iter()
                .chain(state.stable_set.iter())
                .copied()
                .collect();
            for idx in indices {
                let j_s = j_s_values[idx];
                let norm = j_s.norm();
                if norm > eps {
                    state.frames[idx] = j_s / norm;
                }
            }
            state.stable_set.len()
        };

        Ok(StepInfo {
            max_distortion,
            active_set_size: n_active,
            stable_set_size,
            converged,
        })
    }

    /// 実行時にアルゴリズムパラメータを更新（K, lambda, 正則化）。
    fn update_params(&mut self, params: MappingParams) {
        let k = params.k_bound;
        {
            let (_, state) = self.parts_mut();
            state.k_high = 0.1 + 0.9 * k;
            state.k_low = 0.5 + 0.5 * k;
        }

        let (needs_bh, has_bh) = {
            let (_, state) = self.parts();
            let needs_bh = matches!(
                params.regularization,
                RegularizationType::Biharmonic | RegularizationType::Mixed { .. }
            );
            let has_bh = state
                .precomputed
                .as_ref()
                .map_or(false, |p| p.biharmonic_matrix.is_some());
            (needs_bh, has_bh)
        };

        if needs_bh && !has_bh {
            let (ctx, state) = self.parts_mut();
            ensure_biharmonic_matrix_inner(ctx.basis, ctx.domain_bounds, state);
        }

        self.set_params(params);
    }

    /// Strategy 2 事後精緻化（Section 5 "Strategies"）。
    fn refine_strategy2(
        &mut self,
        k_max: f64,
        target_handles: &[Vector2<f64>],
    ) -> Result<strategy::Strategy2Result, AlgorithmError> {
        let (k, c_norm, required_h, current_h) = {
            let (ctx, state) = self.parts();
            let k = ctx.params.k_bound;
            if k_max <= k {
                return Err(AlgorithmError::InvalidInput(format!(
                    "Strategy 2 requires K_max ({}) > K ({})",
                    k_max, k
                )));
            }

            let c_norm = strategy::compute_c_norm(&state.coefficients);
            let required_h = ctx.policy
                .required_h(k, k_max, c_norm, ctx.basis)
                .ok_or_else(|| {
                    AlgorithmError::InvalidInput(
                        "Strategy 2: cannot compute required h (K_max too close to K or c_norm issue)"
                            .into(),
                    )
                })?;
            let current_h = strategy::fill_distance(ctx.domain_bounds, state.grid_width);
            (k, c_norm, required_h, current_h)
        };

        let new_resolution = {
            let (ctx, _state) = self.parts();
            let new_resolution = strategy::resolution_for_h(ctx.domain_bounds, required_h);
            let max_res = ctx.solver_config.max_refinement_resolution;
            if new_resolution > max_res {
                return Err(AlgorithmError::ResolutionExceeded {
                    required: new_resolution,
                    max: max_res,
                });
            }
            new_resolution
        };

        // 必要ならグリッドを再構築
        let needs_rebuild = {
            let (_, state) = self.parts();
            new_resolution > state.grid_width
        };
        if needs_rebuild {
            self.rebuild_grid(new_resolution);
        }

        let mut refinement_steps = 0;
        for _ in 0..strategy::MAX_REFINEMENT_STEPS {
            let info = self.step(target_handles)?;
            refinement_steps += 1;
            if info.converged {
                break;
            }
        }

        let k_max_achieved = {
            let (ctx, state) = self.parts();
            let final_h = strategy::fill_distance(ctx.domain_bounds, state.grid_width);
            let final_omega = strategy::omega(final_h, c_norm, ctx.basis);
            ctx.policy.compute_k_max(k, final_omega).unwrap_or_else(|| {
                warn!(
                    "Strategy 2: cannot guarantee finite K_max \
                     (omega(h)={:.4} >= 1/K={:.4})",
                    final_omega,
                    1.0 / k,
                );
                f64::INFINITY
            })
        };

        Ok(strategy::Strategy2Result {
            required_h,
            required_resolution: new_resolution,
            current_h,
            k_max_achieved,
            c_norm,
            refinement_steps,
        })
    }

    /// 複数点で順方向写像 f(x) = Σ c_i φ_i(x) を評価（Eq. 3）。
    ///
    /// ドメイン空間の点のスライスを受け取り、写像後の位置を返す。
    /// 任意の基底関数タイプに対する CPU レンダリングパスで使用。
    fn evaluate_mapping_at(&self, points: &[Vector2<f64>]) -> Vec<Vector2<f64>> {
        let (ctx, state) = self.parts();
        let c = &state.coefficients;
        let n = ctx.basis.count();
        points
            .iter()
            .map(|&x| {
                let phi = ctx.basis.evaluate(x);
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
    fn grad_uv_at(&self, x: Vector2<f64>) -> (Vector2<f64>, Vector2<f64>) {
        let (ctx, state) = self.parts();
        let (gx, gy) = ctx.basis.gradient(x);
        let c = &state.coefficients;
        let n = ctx.basis.count();

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
    fn j_s_j_a_at(&self, x: Vector2<f64>) -> (Vector2<f64>, Vector2<f64>) {
        let (grad_u, grad_v) = self.grad_uv_at(x);
        distortion::compute_j_s_j_a(grad_u, grad_v)
    }

    /// 点 x で特異値（Σ, σ）を計算（Eq. 20）。
    fn singular_values_at(&self, x: Vector2<f64>) -> (f64, f64) {
        let (j_s, j_a) = self.j_s_j_a_at(x);
        distortion::singular_values(j_s, j_a)
    }

    /// コロケーショングリッドを新しい解像度で再構築（Strategy 2 用）。
    /// 現在の係数を保持。アクティブ集合、フレーム、事前計算データをリセット。
    fn rebuild_grid(&mut self, new_resolution: usize) {
        let (ctx, state) = self.parts_mut();

        let (new_points, new_gw, new_gh) =
            generate_collocation_grid(ctx.domain_bounds, new_resolution);
        let m = new_points.len();
        let new_mask = build_domain_mask(&new_points, ctx.domain);
        let new_frames = vec![Vector2::new(1.0, 0.0); m];

        let fps_k = state.stable_set.len().max(4);
        let new_stable_set =
            active_set::initialize_stable_set(&new_points, fps_k, &new_mask);

        let coefficients = state.coefficients.clone();

        *state = AlgorithmState {
            coefficients,
            collocation_points: new_points,
            active_set: Vec::new(),
            stable_set: new_stable_set,
            frames: new_frames,
            k_high: state.k_high,
            k_low: state.k_low,
            domain_mask: new_mask,
            precomputed: None,
            grid_width: new_gw,
            grid_height: new_gh,
            prev_target_handles: None,
        };
    }
}

// ─────────────────────────────────────────────
// プライベートヘルパー（デフォルトメソッドで使用）
// ─────────────────────────────────────────────

/// 事前計算データに biharmonic 行列（Eq. 31）を構築。
/// トレイトデフォルトメソッドでの &mut self 借用の衝突を避けるための自由関数。
fn ensure_biharmonic_matrix_inner(
    basis: &dyn BasisFunction,
    domain_bounds: &crate::model::types::DomainBounds,
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
    bounds: &crate::model::types::DomainBounds,
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
    domain: Option<&dyn crate::model::domain::Domain>,
) -> Vec<bool> {
    match domain {
        Some(d) => points.iter().map(|pt| d.contains(pt)).collect(),
        None => vec![true; points.len()],
    }
}
