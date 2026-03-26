//! コアアルゴリズム状態: Bevy リソースとして使用するために pgpm-core の MappingBridge をラップ。

use bevy::prelude::*;
use nalgebra::Vector2;
use pgpm_core::mapping::MappingBridge;

/// コアアルゴリズム状態、ソルバーとレンダリングシステムで使用。
#[derive(Resource)]
pub struct AlgorithmState {
    /// pgpm-core の写像インスタンス。ファイナライズ時に作成。
    pub algorithm: Option<Box<dyn MappingBridge>>,
    /// ドメイン（ピクセル）座標でのソースハンドル位置。
    pub source_handles: Vec<Vector2<f64>>,
    /// ドメイン（ピクセル）座標での現在のターゲットハンドル位置。
    pub target_handles: Vec<Vector2<f64>>,
    /// アルゴリズムがステップを実行する必要があるか（ターゲットが変更された）。
    pub needs_solve: bool,
}

impl Default for AlgorithmState {
    fn default() -> Self {
        Self {
            algorithm: None,
            source_handles: Vec::new(),
            target_handles: Vec::new(),
            needs_solve: false,
        }
    }
}

impl AlgorithmState {
    /// 全状態をデフォルトにリセット（画像再読み込みとリセットボタンで使用）。
    pub fn reset(&mut self) {
        self.source_handles.clear();
        self.target_handles.clear();
        self.algorithm = None;
        self.needs_solve = false;
    }

    /// 写像インスタンスを設定し、ターゲットをソース位置に初期化。
    pub fn set_mapping(&mut self, algorithm: Box<dyn MappingBridge>) {
        self.target_handles = self.source_handles.clone();
        self.algorithm = Some(algorithm);
    }
}

/// 元の（変形前の）頂点位置を格納。
///
/// メッシュ生成時に一度だけ作成。以下で使用:
/// - CPU 変形パス: `pixel_positions` を `evaluate_mapping_at()` への入力として
/// - リセット: `world_positions` でメッシュを変形前の状態に復元
#[derive(Resource)]
pub struct OriginalVertexPositions {
    /// ドメイン（ピクセル）座標での頂点位置。
    pub pixel_positions: Vec<Vector2<f64>>,
    /// ワールド座標での頂点位置（元の変形前）。
    pub world_positions: Vec<[f32; 3]>,
}
