//! 歪みポリシートレイトと具象実装。
//!
//! `DistortionPolicy` トレイトは歪みタイプ固有の動作
//! （等長 vs 等角）をカプセル化する:
//! - 歪み値の計算（Section 3）
//! - SOCP制約の構築（Eq. 23/26 vs Eq. 28）
//! - Strategy 2 充填距離の計算（Eq. 14 vs Eq. 15）
//!
//! これは `pub(crate)` — 外部利用者は `PgpmAlgorithm` と
//! ファクトリ関数経由でアクセスする。

use crate::basis::BasisFunction;
use crate::distortion;
use crate::algorithm::strategy;
use crate::model::types::{AlgorithmState, PrecomputedData};
use crate::numerics::solver;

/// SOCP定式化のための歪みタイプ固有の動作。
pub trait DistortionPolicy: Send + Sync {
    /// 特異値から歪み値を計算（Section 3）。
    fn distortion_value(&self, sigma_max: f64, sigma_min: f64) -> f64;

    /// アクティブ点ごとの追加決定変数の数。
    /// 等長: 2（Eq. 23 の t_i, s_i）。等角: 0。
    fn extra_vars_per_active(&self) -> usize;

    /// SOCPに歪み制約を追加（Eq. 23/26 または 28）。
    fn append_constraints(
        &self,
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
    );

    /// Strategy 2: 必要な充填距離 h を計算（Eq. 14 または 15）。
    /// 計算不可能な場合は `None` を返す。
    fn required_h(&self, k: f64, k_max: f64, c_norm: f64, basis: &dyn BasisFunction)
        -> Option<f64>;

    /// Strategy 1: K と omega(h) から K_max を計算（Eq. 11 または 13）。
    /// 単射性が保証できない場合は `None` を返す。
    fn compute_k_max(&self, k: f64, omega_h: f64) -> Option<f64>;
}

/// 等長歪みポリシー: D_iso(x) = max{Σ(x), 1/σ(x)}。
///
/// 制約: Eq. 23a-c, 26。Strategy 2: Eq. 14。
pub struct IsometricPolicy;

impl DistortionPolicy for IsometricPolicy {
    fn distortion_value(&self, sigma_max: f64, sigma_min: f64) -> f64 {
        distortion::isometric_distortion(sigma_max, sigma_min)
    }

    fn extra_vars_per_active(&self) -> usize {
        2 // t_i, s_i（Eq. 23）
    }

    fn append_constraints(
        &self,
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
        solver::append_isometric_constraints(
            state, precomputed, n_basis, n_handles,
            active_indices, n_active, k,
            rows, b, cones,
        );
    }

    fn required_h(&self, k: f64, k_max: f64, c_norm: f64, basis: &dyn BasisFunction)
        -> Option<f64>
    {
        strategy::required_h_isometric(k, k_max, c_norm, basis)
    }

    fn compute_k_max(&self, k: f64, omega_h: f64) -> Option<f64> {
        strategy::compute_k_max_isometric(k, omega_h)
    }
}

/// 等角歪みポリシー: D_conf(x) = Σ(x) / σ(x)。
///
/// 制約: Eq. 28a-b。
/// Strategy 関数: Phase 3 — 等長バージョンにフォールバック。
pub struct ConformalPolicy {
    pub delta: f64,
}

impl DistortionPolicy for ConformalPolicy {
    fn distortion_value(&self, sigma_max: f64, sigma_min: f64) -> f64 {
        distortion::conformal_distortion(sigma_max, sigma_min)
    }

    fn extra_vars_per_active(&self) -> usize {
        0 // 等角制約（Eq. 28）は追加変数不要
    }

    fn append_constraints(
        &self,
        state: &AlgorithmState,
        precomputed: &PrecomputedData,
        n_basis: usize,
        _n_handles: usize,
        active_indices: &[usize],
        _n_active: usize,
        k: f64,
        rows: &mut Vec<Vec<(usize, f64)>>,
        b: &mut Vec<f64>,
        cones: &mut Vec<clarabel::solver::SupportedConeT<f64>>,
    ) {
        solver::append_conformal_constraints(
            state, precomputed, n_basis,
            active_indices, k, self.delta,
            rows, b, cones,
        );
    }

    /// Phase 3: 等角 strategy（Eq. 15）は未実装。
    /// 等長の required_h（Eq. 14）にフォールバック。
    fn required_h(&self, k: f64, k_max: f64, c_norm: f64, basis: &dyn BasisFunction)
        -> Option<f64>
    {
        // Phase 3: 等角用の Eq. 15 を実装
        strategy::required_h_isometric(k, k_max, c_norm, basis)
    }

    /// Phase 3: 等角 K_max（Eq. 13）は未実装。
    /// 等長の K_max（Eq. 11）にフォールバック。
    fn compute_k_max(&self, k: f64, omega_h: f64) -> Option<f64> {
        // Phase 3: 等角用の Eq. 13 を実装
        strategy::compute_k_max_isometric(k, omega_h)
    }
}
