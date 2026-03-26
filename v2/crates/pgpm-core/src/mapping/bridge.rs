//! フロントエンドブリッジトレイト。

use crate::algorithm::strategy;
use crate::mapping::PgpmAlgorithm;
use crate::model::types::{AlgorithmError, DomainBounds, MappingParams, StepInfo};
use nalgebra::Vector2;

/// フロントエンドブリッジ: UI利用者に公開する [`PgpmAlgorithm`] のサブセット。
///
/// `bevy-pgpm` はこのトレイトにのみ依存し、`PgpmAlgorithm` には直接依存しない。
/// 内部メソッド（`coefficients`, `basis`, `grad_uv_at`, `j_s_j_a_at`,
/// `singular_values_at`）はアルゴリズム内部で使用され、
/// このインターフェースには含まれない。
pub trait MappingBridge: Send + Sync {
    /// Algorithm 1: 1ステップを実行する (Section 5)。
    fn step(&mut self, target_handles: &[Vector2<f64>]) -> Result<StepInfo, AlgorithmError>;

    /// 複数の点での前方写像 f(x) = Σ c_i φ_i(x) を評価する (Eq. 3)。
    ///
    /// ドメイン空間の点のスライスを受け取り、写像後の位置を返す。
    /// 任意の基底関数型に対するCPUレンダリングパスで使用。
    fn evaluate_mapping_at(&self, points: &[Vector2<f64>]) -> Vec<Vector2<f64>>;

    /// 実行時にアルゴリズムパラメータを更新する（K, lambda, 正則化）。
    fn update_params(&mut self, params: MappingParams);

    /// Strategy 2 事後細分化 (Section 5 "Strategies")。
    fn refine_strategy2(
        &mut self,
        k_max: f64,
        target_handles: &[Vector2<f64>],
    ) -> Result<strategy::Strategy2Result, AlgorithmError>;

    // ─────────────────────────────────────────
    // クエリメソッド: 読み取り専用の状態検査
    // ─────────────────────────────────────────

    /// 現在のアルゴリズムパラメータ（K, lambda, 正則化）を取得する。
    fn params(&self) -> MappingParams;

    /// コロケーショングリッドの解像度 (width, height) (Section 4)。
    fn grid_resolution(&self) -> (usize, usize);

    /// コロケーション点の総数 |Z| (Section 4)。
    fn num_collocation_points(&self) -> usize;

    /// 基底関数の個数 n (Table 1)。
    fn num_basis_functions(&self) -> usize;

    /// ソースハンドル位置 {p_l} (Eq. 29)。
    fn source_handles(&self) -> Vec<Vector2<f64>>;

    /// ドメイン Ω のバウンディングボックス (Eq. 5)。
    fn domain_bounds(&self) -> DomainBounds;
}

/// ブランケット実装: `PgpmAlgorithm` を実装する型は自動的に `MappingBridge` を満たす。
impl<T: PgpmAlgorithm + ?Sized> MappingBridge for T {
    fn step(&mut self, target_handles: &[Vector2<f64>]) -> Result<StepInfo, AlgorithmError> {
        PgpmAlgorithm::step(self, target_handles)
    }

    fn evaluate_mapping_at(&self, points: &[Vector2<f64>]) -> Vec<Vector2<f64>> {
        PgpmAlgorithm::evaluate_mapping_at(self, points)
    }

    fn update_params(&mut self, params: MappingParams) {
        PgpmAlgorithm::update_params(self, params)
    }

    fn refine_strategy2(
        &mut self,
        k_max: f64,
        target_handles: &[Vector2<f64>],
    ) -> Result<strategy::Strategy2Result, AlgorithmError> {
        PgpmAlgorithm::refine_strategy2(self, k_max, target_handles)
    }

    fn params(&self) -> MappingParams {
        PgpmAlgorithm::params(self)
    }

    fn grid_resolution(&self) -> (usize, usize) {
        PgpmAlgorithm::grid_resolution(self)
    }

    fn num_collocation_points(&self) -> usize {
        PgpmAlgorithm::num_collocation_points(self)
    }

    fn num_basis_functions(&self) -> usize {
        PgpmAlgorithm::num_basis_functions(self)
    }

    fn source_handles(&self) -> Vec<Vector2<f64>> {
        PgpmAlgorithm::source_handles(self)
    }

    fn domain_bounds(&self) -> DomainBounds {
        PgpmAlgorithm::domain_bounds_query(self)
    }
}
