//! Lifecycle systems: image loading/reloading and algorithm stepping.

use bevy::prelude::*;
use bevy::image::Image as BevyImage;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use log::{info, warn, error};
use image::GenericImageView;

use crate::domain::image_loader::extract_contour_from_image;
use crate::rendering::{create_contour_mesh, DeformMaterial, DeformUniform};
use crate::state::{
    AlgorithmState, AppState, DeformationInfo, DeformedImage, ImageInfo, ImagePathConfig,
};

/// System: load (or reload) the image when `ImagePathConfig.needs_reload` is set.
pub fn load_image(
    mut commands: Commands,
    mut path_config: ResMut<ImagePathConfig>,
    mut images: ResMut<Assets<BevyImage>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<DeformMaterial>>,
    existing_image: Query<Entity, With<DeformedImage>>,
    mut algo_state: ResMut<AlgorithmState>,
    mut deform_info: ResMut<DeformationInfo>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    if !path_config.needs_reload {
        return;
    }
    path_config.needs_reload = false;

    let abs_path = &path_config.abs_path;
    info!("Loading image from: {}", abs_path);

    // Load the image with the `image` crate (for dimensions, contour, AND GPU texture).
    // By loading manually and inserting into Assets<Image>, we bypass AssetServer
    // path resolution issues entirely.
    let img = match image::open(abs_path) {
        Ok(img) => img,
        Err(e) => {
            error!("Failed to load image '{}': {}", abs_path, e);
            return;
        }
    };

    let (w, h) = img.dimensions();
    let image_width = w as f32;
    let image_height = h as f32;
    info!("Image dimensions: {}x{}", w, h);

    let contour_data = extract_contour_from_image(abs_path);
    let contour = contour_data.outer;
    let holes = contour_data.holes;

    if contour.is_empty() {
        info!("No contour extracted, using full image domain");
    } else {
        info!(
            "Extracted contour with {} points, {} hole(s)",
            contour.len(),
            holes.len(),
        );
    }

    // Convert to RGBA8 and insert directly into Bevy's Assets<Image>.
    // This avoids AssetServer path resolution issues entirely.
    let rgba = img.to_rgba8();
    let bevy_image = BevyImage::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        rgba.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    );
    let image_handle = images.add(bevy_image);

    commands.insert_resource(ImageInfo {
        width: image_width,
        height: image_height,
        handle: image_handle.clone(),
        contour: contour.clone(),
        holes: holes.clone(),
    });

    // Remove previous image entity if reloading
    for entity in existing_image.iter() {
        commands.entity(entity).despawn();
    }

    // Reset state on image change
    algo_state.reset();
    *deform_info = DeformationInfo::default();
    next_state.set(AppState::Setup);

    // Create mesh (Delaunay triangulation)
    let grid_mesh = create_contour_mesh(
        Vec2::new(image_width, image_height),
        UVec2::new(200, 200),
        &contour,
        &holes,
    );

    let mesh_handle = meshes.add(grid_mesh);

    let material_handle = materials.add(DeformMaterial {
        source_texture: image_handle,
        params: DeformUniform::identity(image_width, image_height),
    });

    commands.spawn((
        Mesh2d(mesh_handle),
        MeshMaterial2d(material_handle),
        Transform::from_xyz(0.0, 0.0, 0.0),
        DeformedImage,
    ));
}

/// System: run one Algorithm step if needed.
///
/// Run condition: `in_state(AppState::Deforming)` via DeformingSet.
pub fn update_deformation(
    mut algo_state: ResMut<AlgorithmState>,
    mut deform_info: ResMut<DeformationInfo>,
) {
    // Keep iterating while dragging OR while the algorithm hasn't converged.
    // Convergence is determined by pgpm-core (Algorithm 1: max_distortion ≤ K
    // and active set stable).
    let needs_more = algo_state.needs_solve
        || (deform_info.step_count > 0 && !deform_info.converged);

    if !needs_more {
        return;
    }

    let targets: Vec<nalgebra::Vector2<f64>> = algo_state.target_handles.clone();

    // Run exactly one Algorithm 1 step per frame (SOCP is blocking).
    if let Some(ref mut algo) = algo_state.algorithm {
        match algo.step(&targets) {
            Ok(step_info) => {
                deform_info.max_distortion = step_info.max_distortion;
                deform_info.active_set_size = step_info.active_set_size;
                deform_info.stable_set_size = step_info.stable_set_size;
                deform_info.converged = step_info.converged;
                deform_info.step_count += 1;
            }
            Err(e) => {
                warn!("SOCP solve failed: {:?}", e);
            }
        }
    }

    algo_state.needs_solve = false;
}


