//! Coordinate conversion between pixel space and Bevy world space.
//!
//! Pixel space: origin at top-left, x right, y down.
//! World space: origin at center, x right, y up.

use bevy::prelude::*;

/// Converts between pixel coordinates and Bevy world coordinates.
pub struct ImageCoords {
    pub width: f32,
    pub height: f32,
}

impl ImageCoords {
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    /// Pixel (x_right, y_down) to World (x_right, y_up), centered.
    pub fn pixel_to_world(&self, px: f32, py: f32) -> Vec2 {
        Vec2::new(px - self.width * 0.5, self.height * 0.5 - py)
    }

    /// World (x_right, y_up) to Pixel (x_right, y_down).
    pub fn world_to_pixel(&self, world: Vec2) -> (f32, f32) {
        (world.x + self.width * 0.5, self.height * 0.5 - world.y)
    }
}
