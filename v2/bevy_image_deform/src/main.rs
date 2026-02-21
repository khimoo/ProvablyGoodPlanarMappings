/*
 * Bevy Image Deformer - Provably Good Planar Mappings
 *
 * Architecture:
 * - The deformation algorithm is MESHLESS (uses Gaussian RBF basis functions)
 * - Rust extracts contour from PNG image
 * - Python computes deformation coefficients with provable distortion bounds
 * - Python sends mapping parameters (coefficients, centers, s) to Rust
 * - Rust evaluates f(x) = Σ c_i * φ_i(x) for each pixel to render deformed image
 *
 * Workflow:
 * 1. Setup Mode: User clicks to add control points
 * 2. Finalize: Build solver with basis functions centered at control points
 * 3. Deform Mode: User drags control points
 *    - onMouseDown: Python precomputes matrices (start_drag)
 *    - onMouseMove: Python solves and returns mapping parameters (update_drag)
 *    - onMouseUp: Python verifies distortion bounds (end_drag + Strategy 2)
 * 4. Rendering: Rust evaluates f(x) for each pixel using the mapping parameters
 */

use bevy::{
    prelude::*,
    render::camera::OrthographicProjection,
    sprite::{Material2dPlugin, MeshMaterial2d},
};
use image::GenericImageView;

use bevy_image_deform::{
    state::{AppMode, ControlPoints, MappingParameters, DeformedImage, ModeText, MainCamera},
    python::{PythonChannels, PyCommand, PyResult, python_thread_loop},
    image::{ImageData, extract_contour_from_image},
    rendering::{DeformMaterial, render_deformed_image, create_grid_mesh},
    input::handle_input,
    ui::{update_ui_text, draw_control_points},
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy Deformer - Setup & Deform Modes".into(),
                resolution: (800.0, 800.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(Material2dPlugin::<DeformMaterial>::default())
        .init_state::<AppMode>()
        .init_resource::<ControlPoints>()
        .init_resource::<MappingParameters>()
        .add_systems(Startup, setup)
        .add_systems(Update, (
            setup_camera_scale,
            handle_input,
            receive_python_results,
            render_deformed_image,
        ))
        .add_systems(Update, (
            draw_control_points,
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
    commands.spawn((Camera2d::default(), MainCamera));

    commands.spawn((
        Text::new("Mode: Setup\n[Click] Add Point\n[Enter] Start Deform\n[R] Reset"),
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        ModeText,
    ));

    let (tx_cmd, rx_cmd) = tokio::sync::mpsc::channel::<PyCommand>(10);
    let (tx_res, rx_res) = crossbeam_channel::bounded::<PyResult>(5);

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(python_thread_loop(rx_cmd, tx_res));
    });

    commands.insert_resource(PythonChannels {
        tx_command: tx_cmd.clone(),
        rx_result: rx_res,
    });

    let image_path = "texture.png";
    let image_handle = asset_server.load(image_path);

    let full_image_path = format!("assets/{}", image_path);
    let (image_width, image_height) = match image::open(&full_image_path) {
        Ok(img) => {
            let (w, h) = img.dimensions();
            println!("Loaded image: {}x{}", w, h);
            (w as f32, h as f32)
        }
        Err(e) => {
            eprintln!("Failed to load image for size detection: {}, using default 512x512", e);
            (512.0, 512.0)
        }
    };

    let contour = extract_contour_from_image(&full_image_path);
    if contour.is_empty() {
        println!("No contour extracted, using full image domain");
    } else {
        println!("Extracted contour with {} points", contour.len());
    }

    commands.insert_resource(ImageData {
        width: image_width,
        height: image_height,
        handle: image_handle.clone(),
        contour: contour.clone(),
    });

    // Create a fine grid mesh (100x100 subdivisions)
    let grid_mesh = create_grid_mesh(
        Vec2::new(image_width, image_height),
        UVec2::new(100, 100),
    );
    let mesh_handle = meshes.add(grid_mesh);

    let mut initial_params = bevy_image_deform::rendering::material::DeformUniform::default();
    initial_params.image_width = image_width;
    initial_params.image_height = image_height;
    initial_params.n_rbf = 0;
    // s_param が 0 だと計算時にゼロ除算やNaNが発生する可能性があるため適当な値を入れておく
    initial_params.s_param = 50.0;

    // 初期状態を恒等写像(Identity Mapping)に設定
    // f_x(x,y) = 0*1 + 1*x + 0*y  => x
    // f_y(x,y) = 0*1 + 0*x + 1*y  => y

    // [0]: 定数項 (shift)
    initial_params.coeffs[0].x = 0.0;
    initial_params.coeffs[0].y = 0.0;

    // [1]: xの係数
    initial_params.coeffs[1].x = 1.0;
    initial_params.coeffs[1].y = 0.0;

    // [2]: yの係数
    initial_params.coeffs[2].x = 0.0;
    initial_params.coeffs[2].y = 1.0;

    let material_handle = materials.add(DeformMaterial {
        source_texture: image_handle.clone(),
        params: initial_params,
    });

    commands.spawn((
        Mesh2d(mesh_handle),
        MeshMaterial2d(material_handle),
        Transform::from_xyz(0.0, 0.0, 0.0),
        DeformedImage,
    ));

    let epsilon = 50.0;
    println!(
        "Initializing domain: {}x{}, epsilon={}",
        image_width, image_height, epsilon
    );
    let _ = tx_cmd.try_send(PyCommand::InitializeDomain {
        width: image_width,
        height: image_height,
        epsilon,
    });

    if !contour.is_empty() {
        let _ = tx_cmd.try_send(PyCommand::SetContour { contour });
    }
}

fn receive_python_results(
    channels: Res<PythonChannels>,
    mut mapping_params: ResMut<MappingParameters>,
    mut next_state: ResMut<NextState<AppMode>>,
    state: Res<State<AppMode>>,
) {
    while let Ok(res) = channels.rx_result.try_recv() {
        match res {
            PyResult::DomainInitialized => {
                println!("Domain initialized");
            }
            PyResult::SetupFinalized => {
                println!("Setup finalized, switching to Deform mode");
                if *state.get() == AppMode::Finalizing {
                    next_state.set(AppMode::Deform);
                }
            }
            PyResult::BasisFunctionParameters {
                coefficients,
                centers,
                s_param,
                n_rbf,
            } => {
                println!("Received basis function parameters: {} RBFs", n_rbf);

                mapping_params.coefficients = coefficients;
                mapping_params.centers = centers;
                mapping_params.s_param = s_param;
                mapping_params.n_rbf = n_rbf;
                mapping_params.is_valid = true;
            }
        }
    }
}

fn setup_camera_scale(
    image_data: Option<Res<ImageData>>,
    windows: Query<&Window>,
    mut camera_q: Query<&mut OrthographicProjection, (With<MainCamera>, Without<Camera2d>)>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }

    let Some(image_data) = image_data else {
        return;
    };

    let Ok(window) = windows.get_single() else {
        return;
    };

    let Ok(mut projection) = camera_q.get_single_mut() else {
        return;
    };

    let window_width = window.width();
    let window_height = window.height();
    let image_width = image_data.width;
    let image_height = image_data.height;

    let margin = 0.9;
    let scale_x = (window_width * margin) / image_width;
    let scale_y = (window_height * margin) / image_height;
    let scale = scale_x.min(scale_y);

    projection.scale = 1.0 / scale;

    println!(
        "Camera scale adjusted: window={}x{}, image={}x{}, scale={}",
        window_width, window_height, image_width, image_height, projection.scale
    );

    *done = true;
}
