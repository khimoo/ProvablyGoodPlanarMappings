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
//! 1. Setup mode: click to place control handles, then press Space.
//! 2. Deform mode: drag handles to deform the image.

use bevy::{
    prelude::*,
    render::camera::OrthographicProjection,
    sprite::{Material2dPlugin, MeshMaterial2d},
};
use image::GenericImageView;

use bevy_pgpm::{
    deform::update_deform_material,
    image::extract_contour_from_image,
    input::{handle_input, update_deformation},
    rendering::{create_contour_mesh, DeformMaterial, DeformUniform, RBFCoeff},
    state::*,
    ui::{draw_handles, update_ui_text},
};

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
        .add_systems(Startup, setup)
        .add_systems(Update, (
            setup_camera_scale,
            handle_input,
            update_deformation,
            update_deform_material,
        ))
        .add_systems(Update, (
            draw_handles,
            update_ui_text,
        ))
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<DeformMaterial>>,
) {
    // Camera
    commands.spawn((Camera2d::default(), MainCamera));

    // UI text
    commands.spawn((
        Text::new("Mode: SETUP\n[Click] Add handle\n[Space] Start deforming\n[R] Reset"),
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        InfoText,
    ));

    // Load image
    let image_path = "texture.png";
    let image_handle = asset_server.load(image_path);

    let full_image_path = format!("assets/{}", image_path);
    let (image_width, image_height) = match image::open(&full_image_path) {
        Ok(img) => {
            let (w, h) = img.dimensions();
            info!("Loaded image: {}x{}", w, h);
            (w as f32, h as f32)
        }
        Err(e) => {
            warn!("Failed to load image: {}, using default 512x512", e);
            (512.0, 512.0)
        }
    };

    // Extract contour for non-rectangular images
    let contour = extract_contour_from_image(&full_image_path);
    if contour.is_empty() {
        info!("No contour extracted, using full image domain");
    } else {
        info!("Extracted contour with {} points", contour.len());
    }

    commands.insert_resource(ImageInfo {
        width: image_width,
        height: image_height,
        handle: image_handle.clone(),
        contour: contour.clone(),
    });

    // Create mesh (Delaunay triangulation)
    let grid_mesh = create_contour_mesh(
        Vec2::new(image_width, image_height),
        UVec2::new(200, 200),
        &contour,
    );
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

/// One-shot system: scale camera to fit the image in the window.
fn setup_camera_scale(
    image_info: Option<Res<ImageInfo>>,
    windows: Query<&Window>,
    mut camera_q: Query<&mut OrthographicProjection, With<MainCamera>>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }

    let Some(image_info) = image_info else { return };
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

    *done = true;
}
