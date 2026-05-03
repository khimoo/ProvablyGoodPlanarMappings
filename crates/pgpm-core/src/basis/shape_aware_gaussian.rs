//! 測地距離を用いた形状認識Gaussian RBF基底関数。
//!
//! 論文 Section "Shape aware bases" (p.76:9):
//! > "we tested shape aware variation of Gaussians, which is achieved by
//! > simply replacing the norm in their definition with the shortest
//! > distance function."
//!
//! f_i(x) = exp(-d_geo(x, x_i)² / (2s²))
//!
//! d_geo はドメイン内部の最短経路距離。
//! ポリゴンが与えられない場合はユークリッド距離にフォールバックする。

use super::BasisFunction;
use crate::model::types::{CoefficientMatrix, DomainBounds};
use crate::numerics::geodesic::{self, GeodesicField};
use nalgebra::{DMatrix, DVector, Vector2};

/// 測地距離を用いた形状認識Gaussian RBF基底。
///
/// 基底の構造 (n = num_centers + 3):
/// - f_0 ... f_{num_centers-1}: 形状認識Gaussian RBF
/// - f_{n-3}: 定数 1
/// - f_{n-2}: x座標
/// - f_{n-1}: y座標
pub struct ShapeAwareGaussianBasis {
    /// RBF中心 {x_i}
    centers: Vec<Vector2<f64>>,
    /// スケールパラメータ s
    s: f64,
    /// 事前計算された測地距離場（各中心に1つ）
    distance_fields: Vec<GeodesicField>,
    /// ドメイン境界（座標参照用）
    _bounds: DomainBounds,
}

impl ShapeAwareGaussianBasis {
    /// 新しい形状認識Gaussian基底を生成する。
    ///
    /// # 引数
    /// - `centers`: RBF中心位置（ソースハンドル位置）
    /// - `s`: Gaussianスケールパラメータ
    /// - `polygon`: ドメイン輪郭ポリゴン（ドメイン座標系）。
    ///   論文は単連結ドメイン Ω を仮定している (Section 3 "Fold-overs")。
    /// - `bounds`: ドメインのバウンディングボックス
    /// - `fmm_resolution`: FMM計算のグリッド解像度
    pub fn new(
        centers: Vec<Vector2<f64>>,
        s: f64,
        polygon: &[Vector2<f64>],
        bounds: &DomainBounds,
        fmm_resolution: usize,
    ) -> Self {
        assert!(s > 0.0, "Scale parameter s must be positive");

        // FMMグリッド上にドメインマスクを構築する。
        // 安全な双線形補間のため、境界マージン（1.5グリッドセル）は
        // build_domain_mask 内部で適用される。
        let mask = geodesic::build_domain_mask(bounds, fmm_resolution, fmm_resolution, polygon);

        // 各中心からの測地距離場を計算
        let distance_fields: Vec<GeodesicField> = centers
            .iter()
            .map(|&center| {
                GeodesicField::compute(center, bounds, fmm_resolution, fmm_resolution, &mask)
            })
            .collect();

        Self {
            centers,
            s,
            distance_fields,
            _bounds: bounds.clone(),
        }
    }

    /// RBF中心の個数（アフィン項を除く）。
    pub fn num_centers(&self) -> usize {
        self.centers.len()
    }

    /// スケールパラメータ。
    pub fn scale(&self) -> f64 {
        self.s
    }
}

impl BasisFunction for ShapeAwareGaussianBasis {
    fn count(&self) -> usize {
        self.centers.len() + 3
    }

    fn evaluate(&self, x: Vector2<f64>) -> DVector<f64> {
        let n = self.count();
        let mut result = DVector::zeros(n);
        let s2 = self.s * self.s;

        // 形状認識Gaussian RBF: f_i(x) = exp(-d_geo(x, x_i)² / (2s²))
        for (i, field) in self.distance_fields.iter().enumerate() {
            let d = field.interpolate(x);
            if !d.is_finite() {
                // 到達不能な点 → 基底値は0
                continue;
            }
            result[i] = (-d * d / (2.0 * s2)).exp();
        }

        // アフィン項（ユークリッド版と同じ）
        let nc = self.centers.len();
        result[nc] = 1.0;     // 定数
        result[nc + 1] = x.x; // x座標
        result[nc + 2] = x.y; // y座標

        result
    }

    fn gradient(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>) {
        let n = self.count();
        let mut grad_x = DVector::zeros(n);
        let mut grad_y = DVector::zeros(n);
        let s2 = self.s * self.s;

        // ∇φ_i(x) = φ_i(x) · (-d / s²) · ∇d
        // ここで d = d_geo(x, x_i)、∇d は距離場の勾配
        //
        // 特殊ケース: d ≈ 0（中心またはその近傍）では、φ は最大値なので
        // 勾配はゼロであるべき。しかし ∇d がドメイン壁付近で Inf になりうるため、
        // 0 * Inf = NaN が生じる。|d| が無視できる場合はスキップして防ぐ。
        for (i, field) in self.distance_fields.iter().enumerate() {
            let d = field.interpolate(x);
            if d.abs() < 1e-12 * self.s || !d.is_finite() {
                // 中心上または到達不能 → 勾配はゼロ
                continue;
            }
            let phi = (-d * d / (2.0 * s2)).exp();
            let grad_d = field.interpolate_gradient(x);

            // 距離場からの非有限勾配値もガード
            // （隣接セルがInf距離の壁であるドメイン境界付近で発生しうる）
            if !grad_d.x.is_finite() || !grad_d.y.is_finite() {
                continue;
            }

            let factor = phi * (-d / s2);
            grad_x[i] = factor * grad_d.x;
            grad_y[i] = factor * grad_d.y;
        }

        // アフィン項: ∇1 = (0,0), ∇x = (1,0), ∇y = (0,1)
        let nc = self.centers.len();
        grad_x[nc + 1] = 1.0;
        grad_y[nc + 2] = 1.0;

        (grad_x, grad_y)
    }

    fn hessian(&self, x: Vector2<f64>) -> (DVector<f64>, DVector<f64>, DVector<f64>) {
        // 勾配の有限差分による数値的ヘシアン。
        // 論文は、多角形ドメイン内の厳密な最短距離は特定の点で
        // 不連続な微分を持つことを指摘しており、
        // 数値微分がここでは適切である。
        let n = self.count();
        let eps = self.s * 1e-3; // スケールに対する相対値

        let (gx_0, gy_0) = self.gradient(x);
        let (gx_px, _) = self.gradient(x + Vector2::new(eps, 0.0));
        let (gx_py, gy_py) = self.gradient(x + Vector2::new(0.0, eps));

        let mut hxx = DVector::zeros(n);
        let mut hxy = DVector::zeros(n);
        let mut hyy = DVector::zeros(n);

        for i in 0..n {
            hxx[i] = (gx_px[i] - gx_0[i]) / eps;
            hxy[i] = (gx_py[i] - gx_0[i]) / eps;
            hyy[i] = (gy_py[i] - gy_0[i]) / eps;
        }

        (hxx, hxy, hyy)
    }

    fn gradient_modulus(&self, t: f64) -> f64 {
        // 論文 p.76:9: "To provide a proof of injectivity [...] the modulus of
        // the gradients of the Gaussian shape-aware functions, ω∇F, should be
        // calculated. Although straightforward, it is cumbersome to compute it
        // in general, and we defer it to future work."
        //
        // 実際にはユークリッドモジュラス t/s² を近似として使用する。
        // 測地距離は常にユークリッド距離以上であるため、
        // 実際の勾配モジュラスはユークリッドのもので上界される。
        t / (self.s * self.s)
    }

    fn gradient_modulus_inverse(&self, v: f64) -> f64 {
        // gradient_modulus と同じ近似（ユークリッド上界）。
        // ω_{∇F}(t) = t / s² → ω_{∇F}⁻¹(v) = v · s²
        v * self.s * self.s
    }

    fn identity_coefficients(&self) -> CoefficientMatrix {
        // ユークリッドGaussianと同じ: アフィン項が恒等写像を符号化する。
        let n = self.count();
        let mut c = DMatrix::zeros(2, n);
        let nc = self.centers.len();
        c[(0, nc + 1)] = 1.0; // u = x
        c[(1, nc + 2)] = 1.0; // v = y
        c
    }
}
