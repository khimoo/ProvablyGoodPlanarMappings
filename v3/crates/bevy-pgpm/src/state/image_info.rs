//! Image-related resources: loaded image data and path configuration.

use bevy::prelude::*;

use crate::domain::coords::ImageCoords;

/// Data about the loaded image.
#[derive(Resource)]
pub struct ImageInfo {
    pub width: f32,
    pub height: f32,
    pub handle: Handle<Image>,
    /// Outer boundary contour in pixel coordinates (empty = full rectangle).
    pub contour: Vec<(f32, f32)>,
    /// Interior hole contours in pixel coordinates.
    pub holes: Vec<Vec<(f32, f32)>>,
}

impl ImageInfo {
    /// Create an `ImageCoords` helper from this image's dimensions.
    pub fn coords(&self) -> ImageCoords {
        ImageCoords::new(self.width, self.height)
    }
}

/// Configuration for the image file path.
///
/// The `abs_path` is used for `image::open()` (contour extraction, dimension query).
/// The `bevy_path` is used for `AssetServer::load()` (GPU texture).
#[derive(Resource)]
pub struct ImagePathConfig {
    /// Absolute filesystem path to the image file.
    pub abs_path: String,
    /// Whether the image needs to be (re)loaded.
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
