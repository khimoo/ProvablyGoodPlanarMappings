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
    render::{
        camera::{ClearColorConfig, OrthographicProjection, Viewport},
        view::RenderLayers,
    },
    sprite::Material2dPlugin,
    window::WindowResized,
};

use bevy_pgpm::{
    input::handle_input,
    lifecycle::{load_image, update_deformation},
    rendering::{
        update_deform_material, cpu_update_mesh_positions,
        is_shape_aware_basis, DeformMaterial,
    },
    state::*,
    ui,
};

/// Resolve the default image path.
/// Expects texture.png in the crate's own assets/ directory.
fn default_image_abs_path() -> String {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set (run via `cargo run`)");
    let p = std::path::PathBuf::from(&manifest)
        .join("assets")
        .join("texture.png");
    assert!(
        p.exists(),
        "Default image not found: {}\n\
         Place a texture.png in the crates/bevy-pgpm/assets/ directory.",
        p.display(),
    );
    let path = p.to_string_lossy().into_owned();
    info!("Using image: {}", path);
    path
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "PGPM - Provably Good Planar Mappings".into(),
                resolution: bevy::window::WindowResolution::new(1280.0, 800.0),
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
            update_deform_material.run_if(not(is_shape_aware_basis)),
            cpu_update_mesh_positions.run_if(is_shape_aware_basis),
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

/// Startup: spawn scene camera and UI camera.
///
/// Two cameras are used to tile the window:
///   - Scene camera (MainCamera): viewport restricted to the left area,
///     renders Mesh2d entities on default render layer 0.
///   - UI camera: covers the full window, on render layer 1 (no scene
///     entities) so it only renders the UI overlay via IsDefaultUiCamera.
fn setup_camera(mut commands: Commands) {
    // Scene camera: default render layer 0, viewport set by setup_camera_viewport.
    commands.spawn((Camera2d::default(), MainCamera));

    // UI camera: render layer 1 (empty) prevents Mesh2d double-rendering.
    // IsDefaultUiCamera directs all UI nodes to this camera.
    commands.spawn((
        Camera2d::default(),
        Camera {
            order: 1,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        RenderLayers::layer(1),
        IsDefaultUiCamera,
    ));
}

/// System: set scene camera viewport to the area left of the UI panel and
/// scale the projection to fit the image. Re-runs on ImageInfo change or
/// window resize.
fn setup_camera_scale(
    image_info: Option<Res<ImageInfo>>,
    windows: Query<&Window>,
    mut camera_q: Query<(&mut Camera, &mut OrthographicProjection), With<MainCamera>>,
    mut resize_events: EventReader<WindowResized>,
) {
    let Some(image_info) = image_info else { return };

    let resized = resize_events.read().last().is_some();
    if !image_info.is_changed() && !resized {
        return;
    }

    let Ok(window) = windows.get_single() else { return };
    let Ok((mut camera, mut projection)) = camera_q.get_single_mut() else { return };

    // Compute viewport in physical pixels (required by Bevy's Viewport).
    let scale_factor = window.scale_factor();
    let physical_w = (window.width() * scale_factor) as u32;
    let physical_h = (window.height() * scale_factor) as u32;
    let panel_physical = (bevy_pgpm::ui::PANEL_WIDTH * scale_factor) as u32;
    let viewport_w = physical_w.saturating_sub(panel_physical).max(1);

    camera.viewport = Some(Viewport {
        physical_position: UVec2::ZERO,
        physical_size: UVec2::new(viewport_w, physical_h),
        ..default()
    });

    // Scale projection to fit the image within the viewport with margin.
    let logical_w = viewport_w as f32 / scale_factor;
    let logical_h = physical_h as f32 / scale_factor;

    let margin = 0.9;
    let scale_x = (logical_w * margin) / image_info.width;
    let scale_y = (logical_h * margin) / image_info.height;
    let scale = scale_x.min(scale_y);

    projection.scale = 1.0 / scale;

    info!(
        "Camera viewport: {}x{} physical, image: {}x{}, scale: {}",
        viewport_w, physical_h, image_info.width, image_info.height, projection.scale
    );
}
