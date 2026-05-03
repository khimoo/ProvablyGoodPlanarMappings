//! ピクセル空間と Bevy ワールド空間の座標変換。
//!
//! ピクセル空間: 原点は左上、x は右、y は下。
//! ワールド空間: 原点は中央、x は右、y は上。

use bevy::prelude::*;

/// ピクセル座標と Bevy ワールド座標を変換。
pub struct ImageCoords {
    pub width: f32,
    pub height: f32,
}

impl ImageCoords {
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    /// ピクセル (x右, y下) から ワールド (x右, y上) へ、中央原点。
    pub fn pixel_to_world(&self, px: f32, py: f32) -> Vec2 {
        Vec2::new(px - self.width * 0.5, self.height * 0.5 - py)
    }

    /// ワールド (x右, y上) から ピクセル (x右, y下) へ。
    pub fn world_to_pixel(&self, world: Vec2) -> (f32, f32) {
        (world.x + self.width * 0.5, self.height * 0.5 - world.y)
    }
}
