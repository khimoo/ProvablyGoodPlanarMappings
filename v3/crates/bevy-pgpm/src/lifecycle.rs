//! Lifecycle systems: image loading/reloading and algorithm stepping.

use bevy::prelude::*;
use bevy::sprite::MeshMaterial2d;
use image::GenericImageView;

use crate::domain::coords::ImageCoords;
use crate::domain::image_loader::extract_contour_from_image;
use crate::rendering::{
    create_contour_mesh, DeformMaterial, DeformUniform, OriginalVertexPositions,
};
use crate::state::{
    AlgorithmState, AppState, DeformationInfo, DeformedImage, ImageInfo, ImagePathConfig,
};

/// System: load (or reload) the image when `ImagePathConfig.needs_reload` is set.
pub fn load_image(
    mut commands: Commands,
    mut path_config: ResMut<ImagePathConfig>,
    asset_server: Res<AssetServer>,
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

    let Some((image_width, image_height)) = load_image_dimensions(abs_path) else {
        return;
    };
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

    let image_handle: Handle<Image> = asset_server.load(abs_path.to_string());

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

    // Store original pixel-space vertex positions for CPU deformation path.
    let pixel_positions = extract_pixel_positions(&grid_mesh, image_width, image_height);
    commands.insert_resource(OriginalVertexPositions {
        positions: pixel_positions,
    });

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

// Private helpers

fn load_image_dimensions(abs_path: &str) -> Option<(f32, f32)> {
    match image::open(abs_path) {
        Ok(img) => {
            let (w, h) = img.dimensions();
            info!("Image dimensions: {}x{}", w, h);
            Some((w as f32, h as f32))
        }
        Err(e) => {
            error!("Failed to load image '{}': {}", abs_path, e);
            None
        }
    }
}

fn extract_pixel_positions(mesh: &Mesh, width: f32, height: f32) -> Vec<[f32; 2]> {
    let coords = ImageCoords::new(width, height);
    let Some(attr) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
        error!("Mesh missing ATTRIBUTE_POSITION — cannot extract vertex positions");
        return vec![];
    };
    let Some(positions) = attr.as_float3() else {
        error!("Mesh ATTRIBUTE_POSITION is not Float32x3 — cannot extract vertex positions");
        return vec![];
    };
    positions
        .iter()
        .map(|[bx, by, _]| {
            let (px, py) = coords.world_to_pixel(Vec2::new(*bx, *by));
            [px, py]
        })
        .collect()
}
