//! アプリケーション状態とリソース。
//!
//! AppState FSM、マーカーコンポーネントを定義し、便利な
//! `use crate::state::*` アクセスのために全サブモジュール型を再エクスポート。

pub mod algorithm;
pub mod display_info;
pub mod image_info;
pub mod interaction;
pub mod params;

pub use algorithm::{AlgorithmState, OriginalVertexPositions};
pub use display_info::DeformationInfo;
pub use image_info::{ImageInfo, ImagePathConfig};
pub use interaction::DragState;
pub use params::{AlgoParams, BasisType, RegMode};

use bevy::prelude::*;

/// アプリケーション状態機械。
/// Setup -> Deforming (-> Phase 3 で Verifying)
#[derive(States, Default, Clone, Eq, PartialEq, Hash, Debug)]
pub enum AppState {
    #[default]
    Setup,
    Deforming,
    // Phase 3: Verifying,
}

/// メインカメラのマーカー。
#[derive(Component)]
pub struct MainCamera;

/// 変形画像エンティティのマーカー。
#[derive(Component)]
pub struct DeformedImage;
