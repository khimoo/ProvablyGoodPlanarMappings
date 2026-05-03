//! 基底関数トレイトとその実装。
//!
//! Table 1: 3種の基底関数 (Gaussian, B-Spline, TPS)。
//! それぞれ値、勾配、ヘシアン、勾配モジュラスを提供する。

pub mod gaussian;
pub mod shape_aware_gaussian;

use crate::model::types::CoefficientMatrix;
use nalgebra::{DVector, Vector2};

/// Table 1 の基底関数の抽象化。
///
/// 各実装は以下を提供する:
/// - 値の評価 f_i(x)
/// - 勾配の評価 ∇f_i(x)
/// - ヘシアンの評価 H_{f_i}(x) (biharmonicエネルギー用、Eq. 31)
/// - 勾配モジュラス ω_{∇F}(t) (Table 1、Eq. 9 で使用)
pub trait BasisFunction: Send + Sync {
    /// 基底関数の個数 n
    fn count(&self) -> usize;

    /// 全基底関数について f_i(x) を評価する。
    /// 長さ n の DVector<f64> を返す。
    fn evaluate(&self, x: Vector2<f64>) -> DVector<f64>;

    /// 全基底関数について ∇f_i(x) を評価する。
    /// (∂f_i/∂x, ∂f_i/∂y) を返す。各々長さ n の DVector<f64>。
    fn gradient(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>);

    /// 全基底関数について H_{f_i}(x) を評価する (Eq. 31)。
    /// (∂²f_i/∂x², ∂²f_i/∂x∂y, ∂²f_i/∂y²) を返す。各々長さ n の DVector<f64>。
    fn hessian(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>, DVector<f64>);

    /// Table 1: 勾配モジュラス ω_{∇F}(t)。
    /// Eq. 9 で使用: ω = 2 |||c||| ω_{∇F}
    fn gradient_modulus(&self, t: f64) -> f64;

    /// 勾配モジュラスの逆関数: ω_{∇F}⁻¹(v) = t （ω_{∇F}(t) = v を満たす t）。
    /// Strategy 2 (Eq. 14) で必要な充填距離 h の計算に使用。
    fn gradient_modulus_inverse(&self, v: f64) -> f64;

    /// 恒等写像の係数 c ∈ R^{2×n} （f(x) = x となるもの）。
    /// (J_f = I が全域で成立)
    fn identity_coefficients(&self) -> CoefficientMatrix;
}
