//! bevy-pgpm: Interactive UI for Provably Good Planar Mappings.
//!
//! Phase 2 implementation: Gaussian basis + Isometric distortion only.
//! Uses pgpm-core for the SOCP-based deformation algorithm and Bevy for
//! rendering/input/UI.
//!
//! Usage:
//!   cargo run -p bevy-pgpm
//!
//! Place a texture.png in the assets/ directory.
//! All interactions are via the on-screen control panel:
//! 1. Setup mode: click to place control handles, then click "Start Deforming".
//! 2. Deform mode: drag handles to deform the image.
//! 3. Adjust K bound, regularization type and lambda via the panel buttons.

use bevy::{
    prelude::*,
    render::camera::OrthographicProjection,
    sprite::{Material2dPlugin, MeshMaterial2d},
};
use image::GenericImageView;

use bevy_pgpm::{
    deform::{update_deform_material, cpu_update_mesh_positions},
    image::extract_contour_from_image,
    input::{handle_input, update_deformation},
    rendering::{create_contour_mesh, DeformMaterial, DeformUniform, RBFCoeff},
    state::*,
    ui,
};

/// Resolve the default image path.
/// Looks for texture.png in the crate's own assets/ directory.
fn default_image_abs_path() -> String {
    // Try CARGO_MANIFEST_DIR first (available during `cargo run`)
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = std::path::PathBuf::from(&manifest)
            .join("assets")
            .join("texture.png");
        if p.exists() {
            return p.to_string_lossy().into_owned();
        }
    }
    // Fallback: try relative to cwd
    let candidates = [
        "crates/bevy-pgpm/assets/texture.png",
        "assets/texture.png",
        "texture.png",
    ];
    for c in &candidates {
        if std::path::Path::new(c).exists() {
            return std::fs::canonicalize(c)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| c.to_string());
        }
    }
    // Last resort
    "texture.png".to_string()
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "PGPM v3 — Provably Good Planar Mappings".into(),
                resolution: (900.0, 900.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(Material2dPlugin::<DeformMaterial>::default())
        .init_state::<AppState>()
        .init_resource::<DeformationState>()
        .init_resource::<DeformationInfo>()
        .init_resource::<AlgoParams>()
        .insert_resource(ImagePathConfig::new(default_image_abs_path()))
        .add_systems(Startup, (setup_camera, ui::spawn_control_panel))
        .add_systems(Update, (
            load_image,
            setup_camera_scale,
            handle_input,
            update_deformation.before(update_deform_material),
            update_deform_material,
            cpu_update_mesh_positions,
        ))
        .add_systems(Update, (
            ui::draw_handles,
            ui::button_visuals,
            ui::on_toggle_mode,
            ui::on_reset,
            ui::on_k_bound,
            ui::on_lambda,
            ui::on_reg_mode,
            ui::on_basis_type,
            ui::on_image_path,
            ui::on_k_max,
            ui::on_strategy2,
            ui::update_status_text,
            ui::update_toggle_label,
            ui::update_k_text,
            ui::update_lambda_text,
            ui::update_reg_mode_label,
            ui::update_basis_type_label,
            ui::update_k_max_text,
        ))
        .run();
}

/// Startup: spawn camera only.  Image loading is handled by the `load_image` system.
fn setup_camera(mut commands: Commands) {
    commands.spawn((Camera2d::default(), MainCamera));
}

/// System: load (or reload) the image when `ImagePathConfig.needs_reload` is set.
fn load_image(
    mut commands: Commands,
    mut path_config: ResMut<ImagePathConfig>,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<DeformMaterial>>,
    existing_image: Query<Entity, With<DeformedImage>>,
    mut deform_state: ResMut<DeformationState>,
    mut deform_info: ResMut<DeformationInfo>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    if !path_config.needs_reload {
        return;
    }
    path_config.needs_reload = false;

    let abs_path = &path_config.abs_path;
    info!("Loading image from: {}", abs_path);

    // Open image via `image` crate for dimensions + contour
    let (image_width, image_height) = match image::open(abs_path) {
        Ok(img) => {
            let (w, h) = img.dimensions();
            info!("Image dimensions: {}x{}", w, h);
            (w as f32, h as f32)
        }
        Err(e) => {
            warn!("Failed to load image '{}': {}, using default 512x512", abs_path, e);
            (512.0, 512.0)
        }
    };

    let contour = extract_contour_from_image(abs_path);
    if contour.is_empty() {
        info!("No contour extracted, using full image domain");
    } else {
        info!("Extracted contour with {} points", contour.len());
    }

    // Load texture through Bevy's AssetServer.
    // If the path is absolute, AssetServer can load it directly.
    // Otherwise it resolves relative to the configured asset folder.
    let image_handle: Handle<Image> = asset_server.load(abs_path.to_string());

    commands.insert_resource(ImageInfo {
        width: image_width,
        height: image_height,
        handle: image_handle.clone(),
        contour: contour.clone(),
    });

    // Remove previous image entity if reloading
    for entity in existing_image.iter() {
        commands.entity(entity).despawn();
    }

    // Reset deformation state on image change
    deform_state.source_handles.clear();
    deform_state.target_handles.clear();
    deform_state.algorithm = None;
    deform_state.dragging = false;
    deform_state.dragging_index = None;
    deform_state.needs_solve = false;
    *deform_info = DeformationInfo::default();
    next_state.set(AppState::Setup);

    // Create mesh (Delaunay triangulation)
    let grid_mesh = create_contour_mesh(
        Vec2::new(image_width, image_height),
        UVec2::new(200, 200),
        &contour,
    );

    // Store original pixel-space vertex positions for CPU deformation path.
    // Vertex POSITION is in world-centered coords: bx = px - w/2, by = h/2 - py.
    // Invert to recover pixel coords: px = bx + w/2, py = h/2 - by.
    let pixel_positions: Vec<[f32; 2]> = grid_mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(|attr| attr.as_float3())
        .map(|positions| {
            positions
                .iter()
                .map(|[bx, by, _]| {
                    [bx + image_width * 0.5, image_height * 0.5 - by]
                })
                .collect()
        })
        .unwrap_or_default();
    commands.insert_resource(bevy_pgpm::deform::OriginalVertexPositions {
        positions: pixel_positions,
    });

    let mesh_handle = meshes.add(grid_mesh);

    // Create initial material with identity mapping (affine only)
    let mut initial_params = DeformUniform::default();
    initial_params.image_width = image_width;
    initial_params.image_height = image_height;
    initial_params.n_rbf = 0;
    // Identity: coeffs[0] = const(0,0), coeffs[1] = x(1,0), coeffs[2] = y(0,1)
    initial_params.coeffs[0] = RBFCoeff { x: 0.0, y: 0.0, _padding: Vec2::ZERO };
    initial_params.coeffs[1] = RBFCoeff { x: 1.0, y: 0.0, _padding: Vec2::ZERO };
    initial_params.coeffs[2] = RBFCoeff { x: 0.0, y: 1.0, _padding: Vec2::ZERO };

    let material_handle = materials.add(DeformMaterial {
        source_texture: image_handle,
        params: initial_params,
    });

    commands.spawn((
        Mesh2d(mesh_handle),
        MeshMaterial2d(material_handle),
        Transform::from_xyz(0.0, 0.0, 0.0),
        DeformedImage,
    ));
}

/// System: scale camera to fit the image in the window.
/// Re-runs whenever ImageInfo is inserted or changed (e.g. after image reload).
fn setup_camera_scale(
    image_info: Option<Res<ImageInfo>>,
    windows: Query<&Window>,
    mut camera_q: Query<&mut OrthographicProjection, With<MainCamera>>,
) {
    let Some(image_info) = image_info else { return };
    if !image_info.is_changed() {
        return;
    }

    let Ok(window) = windows.get_single() else { return };
    let Ok(mut projection) = camera_q.get_single_mut() else { return };

    let window_width = window.width();
    let window_height = window.height();
    let image_width = image_info.width;
    let image_height = image_info.height;

    let margin = 0.9;
    let scale_x = (window_width * margin) / image_width;
    let scale_y = (window_height * margin) / image_height;
    let scale = scale_x.min(scale_y);

    projection.scale = 1.0 / scale;

    info!(
        "Camera scale adjusted: window={}x{}, image={}x{}, scale={}",
        window_width, window_height, image_width, image_height, projection.scale
    );
}
