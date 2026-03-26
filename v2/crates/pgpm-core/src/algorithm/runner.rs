//! Algorithm 1 統合。
//!
//! 論文参照: Algorithm 1 (Section 5)
//! このモジュールは証明付き良好な平面写像のデータを保持する
//! 具象 `Algorithm<D>` 構造体を提供する。全てのアルゴリズムロジックは
//! [`PlanarMapping`] トレイトのデフォルトメソッドに存在し、この構造体は
//! 借用分離アクセサのみを提供する。

use crate::algorithm::active_set;
use crate::basis::BasisFunction;
use crate::mapping::planar_mapping::{build_domain_mask, generate_collocation_grid};
use crate::mapping::PlanarMapping;
use crate::model::domain::Domain;
use crate::model::types::*;
use crate::policy::DistortionPolicy;
use nalgebra::Vector2;

/// Algorithm 1: 歪みポリシーでパラメータ化された完全実装。
pub struct Algorithm<D: DistortionPolicy> {
    /// 基底関数 (Table 1)
    basis: Box<dyn BasisFunction>,
    /// アルゴリズムパラメータ (K, lambda, 正則化タイプ)
    params: MappingParams,
    /// 歪みポリシー (isometric または conformal)
    policy: D,
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

impl<D: DistortionPolicy> Algorithm<D> {
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
        policy: D,
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
    // 固有の便利メソッド（トレイトに無い）
    // ─────────────────────────────────────────

    /// 現在のアクティブ集合サイズを取得。
    pub fn active_set_size(&self) -> usize {
        self.state.active_set.len()
    }

    /// コロケーション点を取得。
    pub fn collocation_points(&self) -> &[Vector2<f64>] {
        &self.state.collocation_points
    }
}

// ─────────────────────────────────────────────
// PlanarMapping トレイト実装
// ─────────────────────────────────────────────

impl<D: DistortionPolicy> PlanarMapping for Algorithm<D> {
    // 全てのアルゴリズムメソッド（step, refine_strategy2 等）は
    // PlanarMapping のデフォルト実装を使用。
    // ここでは3つの必須アクセサのみを提供。

    fn parts(&self) -> (MappingContext<'_>, &AlgorithmState) {
        (
            MappingContext {
                basis: self.basis.as_ref(),
                policy: &self.policy,
                params: &self.params,
                source_handles: &self.source_handles,
                domain_bounds: &self.domain_bounds,
                domain: self.domain.as_deref(),
                solver_config: &self.solver_config,
            },
            &self.state,
        )
    }

    fn parts_mut(&mut self) -> (MappingContext<'_>, &mut AlgorithmState) {
        (
            MappingContext {
                basis: self.basis.as_ref(),
                policy: &self.policy,
                params: &self.params,
                source_handles: &self.source_handles,
                domain_bounds: &self.domain_bounds,
                domain: self.domain.as_deref(),
                solver_config: &self.solver_config,
            },
            &mut self.state,
        )
    }

    fn set_params(&mut self, params: MappingParams) {
        self.params = params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::gaussian::GaussianBasis;
    use crate::policy::IsometricPolicy;

    fn make_test_algorithm() -> Algorithm<IsometricPolicy> {
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
            IsometricPolicy,
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

        let (ctx, state) = alg.parts();
        let precomputed = state.precomputed.as_ref().unwrap();
        let distortions = distortion::evaluate_distortion_all(
            &state.coefficients,
            precomputed,
            ctx.policy,
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
            IsometricPolicy,
            domain,
            centers,
            5,
            2,
            Some(Box::new(polygon_domain)),
            SolverConfig::default(),
        );

        // グリッドは 5x5 = 25 点だが、一部はマスクされるべき
        let (_, state) = alg.parts();
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
