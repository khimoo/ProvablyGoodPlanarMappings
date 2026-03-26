//! UI インタラクション状態: ハンドルに対するドラッグ操作。

use bevy::prelude::*;

/// 現在のドラッグ操作を追跡（入力システムとソルバーシステム間の
/// 共有可変アクセスを避けるためアルゴリズム状態から分離）。
#[derive(Resource, Default)]
pub struct DragState {
    /// ドラッグ操作が現在進行中かどうか。
    pub active: bool,
    /// 現在ドラッグ中のハンドルインデックス（存在する場合）。
    pub handle_index: Option<usize>,
}

impl DragState {
    pub fn start(&mut self, index: usize) {
        self.active = true;
        self.handle_index = Some(index);
    }

    pub fn end(&mut self) {
        self.active = false;
        self.handle_index = None;
    }
}
