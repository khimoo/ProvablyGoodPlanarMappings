//! 画像関連リソース: 読み込まれた画像データとパス設定。

use bevy::prelude::*;

use crate::domain::coords::ImageCoords;

/// 読み込まれた画像のデータ。
#[derive(Resource)]
pub struct ImageInfo {
    pub width: f32,
    pub height: f32,
    pub handle: Handle<Image>,
    /// ピクセル座標での外部境界輪郭（空 = 完全矩形）。
    pub contour: Vec<(f32, f32)>,
    /// ピクセル座標での内部穴輪郭。
    pub holes: Vec<Vec<(f32, f32)>>,
}

impl ImageInfo {
    /// この画像の寸法から `ImageCoords` ヘルパーを作成。
    pub fn coords(&self) -> ImageCoords {
        ImageCoords::new(self.width, self.height)
    }
}

/// 画像ファイルパスの設定。
///
/// 画像への絶対ファイルシステムパスを格納。
/// 画像は `image::open()` で読み込まれ、`AssetServer` のパス解決を
/// バイパスして直接 Bevy の `Assets<Image>` に挿入される。
#[derive(Resource)]
pub struct ImagePathConfig {
    /// 画像ファイルへの絶対ファイルシステムパス。
    pub abs_path: String,
    /// 画像を（再）読み込みする必要があるか。
    pub needs_reload: bool,
}

impl Default for ImagePathConfig {
    fn default() -> Self {
        Self {
            abs_path: String::new(),
            needs_reload: true,
        }
    }
}

impl ImagePathConfig {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            abs_path: path.into(),
            needs_reload: true,
        }
    }
}
