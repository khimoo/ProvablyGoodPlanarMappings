//! Image loading and contour extraction.
//!
//! Loads a PNG image, extracts the alpha-channel contour (for non-rectangular
//! images), and resamples it to a manageable number of points.

use imageproc::contours::{find_contours, BorderType};
use image::GenericImageView;
use log::{error, info, warn};

pub const CONTOUR_TARGET_POINTS: usize = 1024;
pub const ALPHA_THRESHOLD: u8 = 128;

/// Extracted contour data: outer boundary and interior holes.
pub struct ContourData {
    /// Outer boundary contour in pixel coordinates.
    /// Empty means "use full rectangle" (fully opaque image).
    pub outer: Vec<(f32, f32)>,
    /// Interior hole contours in pixel coordinates.
    /// Each hole is a closed polygon; points inside a hole are outside the domain.
    pub holes: Vec<Vec<(f32, f32)>>,
}

/// Extract the contour from an image's alpha channel.
///
/// For fully opaque rectangular images, returns empty outer (meaning "use full rect").
/// For images with transparency, returns the longest outer contour and any
/// hole contours that are direct children of it.
pub fn extract_contour_from_image(image_path: &str) -> ContourData {
    let img = match image::open(image_path) {
        Ok(img) => img,
        Err(e) => {
            error!("Failed to load image for contour extraction: {}", e);
            return ContourData { outer: vec![], holes: vec![] };
        }
    };

    let (width, height) = img.dimensions();
    let rgba = img.to_rgba8();

    // Check if image has any transparent pixels at all
    let has_transparency = rgba.pixels().any(|p| p[3] < ALPHA_THRESHOLD);
    if !has_transparency {
        // Fully opaque: use full rectangle, return empty (caller handles this)
        return ContourData { outer: vec![], holes: vec![] };
    }

    let binary_image = imageproc::map::map_pixels(&rgba, |_x, _y, p| {
        if p[3] >= ALPHA_THRESHOLD {
            image::Luma([255u8])
        } else {
            image::Luma([0u8])
        }
    });

    let contours = find_contours::<u32>(&binary_image);

    if contours.is_empty() {
        warn!("No contours found in image despite transparency; using rectangular boundary");
        return ContourData {
            outer: vec![
                (0.0, 0.0),
                (width as f32, 0.0),
                (width as f32, height as f32),
                (0.0, height as f32),
            ],
            holes: vec![],
        };
    }

    // Find the longest Outer contour → main domain boundary
    let (outer_idx, longest_outer) = contours
        .iter()
        .enumerate()
        .filter(|(_, c)| c.border_type == BorderType::Outer)
        .max_by_key(|(_, c)| c.points.len())
        .unwrap();

    let outer_points: Vec<(f32, f32)> = longest_outer
        .points
        .iter()
        .map(|p| (p.x as f32, p.y as f32))
        .collect();

    // Collect Hole contours whose parent is the main outer contour
    let hole_contours: Vec<Vec<(f32, f32)>> = contours
        .iter()
        .filter(|c| c.border_type == BorderType::Hole && c.parent == Some(outer_idx))
        .map(|c| {
            let points: Vec<(f32, f32)> = c.points
                .iter()
                .map(|p| (p.x as f32, p.y as f32))
                .collect();
            resample_contour(&points, CONTOUR_TARGET_POINTS)
        })
        .collect();

    if !hole_contours.is_empty() {
        info!("Found {} hole contour(s) in image", hole_contours.len());
    }

    ContourData {
        outer: resample_contour(&outer_points, CONTOUR_TARGET_POINTS),
        holes: hole_contours,
    }
}

/// Resample a contour to have at most `target_points` uniformly spaced points.
pub fn resample_contour(contour: &[(f32, f32)], target_points: usize) -> Vec<(f32, f32)> {
    if contour.len() <= target_points {
        return contour.to_vec();
    }

    let mut total_length = 0.0f32;
    for i in 0..contour.len() - 1 {
        let dx = contour[i + 1].0 - contour[i].0;
        let dy = contour[i + 1].1 - contour[i].1;
        total_length += (dx * dx + dy * dy).sqrt();
    }

    if total_length == 0.0 {
        return contour.to_vec();
    }

    let target_spacing = total_length / target_points as f32;

    let mut resampled = Vec::new();
    resampled.push(contour[0]);

    let mut accumulated_dist = 0.0f32;
    let mut next_sample_dist = target_spacing;

    for i in 0..contour.len() - 1 {
        let p1 = contour[i];
        let p2 = contour[i + 1];
        let dx = p2.0 - p1.0;
        let dy = p2.1 - p1.1;
        let segment_length = (dx * dx + dy * dy).sqrt();

        if segment_length == 0.0 {
            continue;
        }

        let segment_end_dist = accumulated_dist + segment_length;

        while next_sample_dist <= segment_end_dist && resampled.len() < target_points {
            let t = (next_sample_dist - accumulated_dist) / segment_length;
            let sample_x = p1.0 + dx * t;
            let sample_y = p1.1 + dy * t;
            resampled.push((sample_x, sample_y));
            next_sample_dist += target_spacing;
        }

        accumulated_dist = segment_end_dist;
    }

    if resampled.len() < target_points {
        resampled.push(*contour.last().unwrap());
    }

    resampled
}
