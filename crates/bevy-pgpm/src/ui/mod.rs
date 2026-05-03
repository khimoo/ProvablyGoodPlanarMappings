//! GUI パネル: ボタン、パラメータコントロール、ハンドルギズモ、ステータス表示。
//!
//! キャンバス上でのハンドル配置/ドラッグを除く全てのユーザーインタラクションは
//! 画面上の UI 要素を通じて行われる。キーボードショートカットは無し。

pub mod markers;
pub mod panel;
pub mod actions;
pub mod display;
pub mod gizmos;

pub use panel::spawn_control_panel;
pub use actions::{
    on_toggle_mode, on_reset, on_k_bound, on_lambda,
    on_reg_mode, on_basis_type, on_image_path, on_k_max, on_strategy2,
    on_export,
};
pub use display::{
    button_visuals, update_status_text, update_toggle_label,
    update_k_text, update_lambda_text, update_reg_mode_label,
    update_basis_type_label, update_k_max_text,
};
pub use gizmos::draw_handles;

use bevy::prelude::*;

/// 右側コントロールパネルの論理ピクセル単位での幅。
pub const PANEL_WIDTH: f32 = 220.0;

// UI サブモジュール間で共有される色。
pub(crate) const PANEL_BG: Color = Color::srgba(0.08, 0.08, 0.12, 0.92);
pub(crate) const BTN_NORMAL: Color = Color::srgb(0.25, 0.25, 0.35);
pub(crate) const BTN_HOVERED: Color = Color::srgb(0.35, 0.35, 0.50);
pub(crate) const BTN_PRESSED: Color = Color::srgb(0.50, 0.40, 0.20);
pub(crate) const BTN_TEXT: Color = Color::srgb(0.95, 0.95, 0.95);
pub(crate) const LABEL_COLOR: Color = Color::srgb(0.70, 0.70, 0.70);
pub(crate) const VALUE_COLOR: Color = Color::srgb(0.95, 0.85, 0.40);
