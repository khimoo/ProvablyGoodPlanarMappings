use imageproc::contours::find_contours;
use image::GenericImageView;

pub const CONTOUR_TARGET_POINTS: usize = 1024;
pub const ALPHA_THRESHOLD: u8 = 128;

pub fn extract_contour_from_image(image_path: &str) -> Vec<(f32, f32)> {
    let img = match image::open(image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load image for contour extraction: {}", e);
            return vec![];
        }
    };

    let (width, height) = img.dimensions();
    let rgba = img.to_rgba8();

    println!("Extracting contour from {}x{} image", width, height);

    let binary_image = imageproc::map::map_pixels(&rgba, |_x, _y, p| {
        if p[3] >= ALPHA_THRESHOLD {
            image::Luma([255u8])
        } else {
            image::Luma([0u8])
        }
    });

    let contours = find_contours::<u32>(&binary_image);

    if contours.is_empty() {
        println!("No contours found, using full image rectangle");
        return vec![
            (0.0, 0.0),
            (width as f32, 0.0),
            (width as f32, height as f32),
            (0.0, height as f32),
        ];
    }

    let longest_contour = contours
        .iter()
        .max_by_key(|c| c.points.len())
        .unwrap();

    println!("Found contour with {} points", longest_contour.points.len());

    let contour_points: Vec<(f32, f32)> = longest_contour
        .points
        .iter()
        .map(|p| (p.x as f32, p.y as f32))
        .collect();

    let resampled = resample_contour(&contour_points, CONTOUR_TARGET_POINTS);

    println!("Resampled contour to {} points", resampled.len());

    resampled
}

pub fn resample_contour(contour: &[(f32, f32)], target_points: usize) -> Vec<(f32, f32)> {
    if contour.len() <= target_points {
        return contour.to_vec();
    }

    let mut total_length = 0.0;
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

    let mut accumulated_dist = 0.0;
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
