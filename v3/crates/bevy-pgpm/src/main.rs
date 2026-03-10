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
    sprite::Material2dPlugin,
};

use bevy_pgpm::{
    input::handle_input,
    lifecycle::{load_image, update_deformation},
    rendering::{update_deform_material, cpu_update_mesh_positions, DeformMaterial},
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
            let path = p.to_string_lossy().into_owned();
            info!("Using image: {}", path);
            return path;
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
            let path = std::fs::canonicalize(c)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| c.to_string());
            info!("Using image: {}", path);
            return path;
        }
    }
    // No image found — report all searched paths
    let manifest_path = std::env::var("CARGO_MANIFEST_DIR")
        .map(|m| format!("{}/assets/texture.png", m))
        .unwrap_or_else(|_| "(CARGO_MANIFEST_DIR not set)".into());
    error!(
        "No default image found. Searched:\n  - {}\n  - {}\n  - {}\n  - {}\n\
         Place a texture.png in the crates/bevy-pgpm/assets/ directory.",
        manifest_path, candidates[0], candidates[1], candidates[2],
    );
    // Return the canonical expected path; load_image will handle the missing file
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        std::path::PathBuf::from(manifest)
            .join("assets")
            .join("texture.png")
            .to_string_lossy()
            .into_owned()
    } else {
        candidates[0].to_string()
    }
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
        .init_resource::<AlgorithmState>()
        .init_resource::<DragState>()
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

/// Startup: spawn camera only. Image loading is handled by the `load_image` system.
fn setup_camera(mut commands: Commands) {
    commands.spawn((Camera2d::default(), MainCamera));
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
