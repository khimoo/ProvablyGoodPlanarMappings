//! 調整可能なアルゴリズムパラメータと列挙型。

use bevy::prelude::*;
use pgpm_core::model::types::RegularizationType;

/// 使用する基底関数タイプ (Table 1)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisType {
    /// 標準ユークリッド Gaussian -- GPU パス（頂点シェーダが RBF を評価）。
    Gaussian,
    /// 測地距離を使用する Shape-aware Gaussian -- CPU パス。
    ShapeAwareGaussian,
    // Phase 3: BSpline,
    // Phase 3: TPS,
}

impl BasisType {
    pub fn next(self) -> Self {
        match self {
            BasisType::Gaussian => BasisType::ShapeAwareGaussian,
            BasisType::ShapeAwareGaussian => BasisType::Gaussian,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            BasisType::Gaussian => "Gaussian",
            BasisType::ShapeAwareGaussian => "Shape-Aware",
        }
    }

    /// GPU シェーダがこの基底を直接評価できるか。
    pub fn supports_gpu_eval(self) -> bool {
        match self {
            BasisType::Gaussian => true,
            BasisType::ShapeAwareGaussian => false,
            // Phase 3: BasisType::BSpline => ...,
            // Phase 3: BasisType::TPS => ...,
        }
    }
}

impl std::fmt::Display for BasisType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// ユーザーが選択した正則化モード。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegMode {
    Arap,
    Biharmonic,
    Mixed,
    None,
}

impl RegMode {
    pub fn next(self) -> Self {
        match self {
            RegMode::Arap => RegMode::Biharmonic,
            RegMode::Biharmonic => RegMode::Mixed,
            RegMode::Mixed => RegMode::None,
            RegMode::None => RegMode::Arap,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            RegMode::Arap => "ARAP",
            RegMode::Biharmonic => "Biharmonic",
            RegMode::Mixed => "Mixed",
            RegMode::None => "None",
        }
    }

    /// pgpm-core の RegularizationType に変換。
    pub fn to_core(self, lambda_arap: f64, lambda_bh: f64) -> RegularizationType {
        match self {
            RegMode::Arap => RegularizationType::Arap,
            RegMode::Biharmonic => RegularizationType::Biharmonic,
            RegMode::Mixed => RegularizationType::Mixed {
                lambda_bh,
                lambda_arap,
            },
            RegMode::None => RegularizationType::Arap, // lambda=0 で対応
        }
    }

    /// 有効な lambda_reg: None モードでは 0 に強制。
    pub fn effective_lambda(self, lambda_reg: f64) -> f64 {
        match self {
            RegMode::None => 0.0,
            _ => lambda_reg,
        }
    }
}

impl std::fmt::Display for RegMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// 調整可能なアルゴリズムパラメータ。
#[derive(Resource)]
pub struct AlgoParams {
    pub k_bound: f64,
    pub lambda_reg: f64,
    pub grid_resolution: usize,
    pub fps_k: usize,
    pub epsilon: f64,
    /// 正則化モード (ARAP / Biharmonic / Mixed / None)
    pub reg_mode: RegMode,
    /// Mixed モード用 ARAP 重み
    pub lambda_arap: f64,
    /// Mixed モード用 Biharmonic 重み
    pub lambda_bh: f64,
    /// 基底関数タイプ (Table 1)
    pub basis_type: BasisType,
    /// Strategy 2 の目標 K_max (Eq. 14、k_bound より大きい必要あり)
    pub k_max: f64,
}

impl Default for AlgoParams {
    fn default() -> Self {
        Self {
            k_bound: 3.0,
            lambda_reg: 1e-2,
            grid_resolution: 50,
            fps_k: 8,
            epsilon: 40.0,
            reg_mode: RegMode::Arap,
            lambda_arap: 1.0,
            lambda_bh: 0.1,
            basis_type: BasisType::Gaussian,
            k_max: 6.0, // デフォルト: k_bound * 2.0
        }
    }
}
