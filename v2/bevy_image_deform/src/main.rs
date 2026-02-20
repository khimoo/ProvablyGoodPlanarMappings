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
    render::{
        render_resource::{Extent3d, TextureDimension, TextureFormat, AsBindGroup, ShaderRef},
        mesh::Mesh2d,
    },
    sprite::{Material2d, Material2dPlugin, MeshMaterial2d},
};
use crossbeam_channel::{bounded, Receiver, Sender};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};
use std::env;

// --- デフォルト定数（ウィンドウサイズ用） ---
const WINDOW_WIDTH: f32 = 1000.0;
const WINDOW_HEIGHT: f32 = 800.0;

// --- アプリケーションの状態 (モード) ---
#[derive(States, Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
enum AppMode {
    #[default]
    Setup,   // 点を追加するモード (変形なし)
    Deform,  // 点をドラッグして変形するモード
}

// --- 通信コマンド ---
enum PyCommand {
    InitializeDomain { width: f32, height: f32, epsilon: f32 },
    SetContour { contour: Vec<(f32, f32)> },
    AddControlPoint { index: usize, x: f32, y: f32 },
    FinalizeSetup,
    StartDrag,
    UpdatePoint { control_index: usize, x: f32, y: f32 },
    EndDrag,
    Reset,
}

// --- 通信結果 ---
enum PyResult {
    DomainInitialized,
    MappingParameters {
        coefficients: Vec<Vec<f32>>,  // (2, N+3)
        centers: Vec<Vec<f32>>,        // (N, 2)
        s_param: f32,
        n_rbf: usize,
        image_width: f32,
        image_height: f32,
        inverse_grid: Vec<Vec<Vec<f32>>>,  // (H, W, 2)
        grid_width: usize,
        grid_height: usize,
    },
}

// --- リソース ---
#[derive(Resource)]
struct PythonChannels {
    tx_command: tokio::sync::mpsc::Sender<PyCommand>,
    rx_result: Receiver<PyResult>,
}

#[derive(Resource, Default)]
struct ControlPoints {
    points: Vec<(usize, Vec2)>,  // (control_index, position in world coords)
    dragging_index: Option<usize>,
}

#[derive(Resource, Default)]
struct MappingParameters {
    coefficients: Vec<Vec<f32>>,  // (2, N+3)
    centers: Vec<Vec<f32>>,        // (N, 2) in image coords
    s_param: f32,
    n_rbf: usize,
    image_width: f32,
    image_height: f32,
    inverse_grid: Vec<Vec<Vec<f32>>>,  // (H, W, 2)
    grid_width: usize,
    grid_height: usize,
    is_valid: bool,
}

#[derive(Resource)]
struct ImageData {
    width: f32,
    height: f32,
    handle: Handle<Image>,
}

#[derive(Component)]
struct DeformedImage;

#[derive(Component)]
struct ModeText;

// Custom material for deformation shader
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct DeformMaterial {
    #[texture(0)]
    #[sampler(1)]
    source_texture: Handle<Image>,
    
    #[texture(2)]
    #[sampler(3)]
    inverse_grid_texture: Handle<Image>,
    
    #[uniform(4)]
    grid_size: Vec2,  // (width, height) of inverse grid
}

impl Material2d for DeformMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/deform.wgsl".into()
    }
}

// 輪郭線抽出: アルファチャンネルから境界を検出
fn extract_contour_from_image(image_path: &str) -> Vec<(f32, f32)> {
    use image::GenericImageView;
    
    let img = match image::open(image_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Failed to load image for contour extraction: {}", e);
            return vec![];
        }
    };
    
    let (width, height) = img.dimensions();
    let rgba = img.to_rgba8();
    
    // 簡単な輪郭抽出: アルファ値が閾値以上のピクセルの境界を見つける
    let alpha_threshold = 128;
    let mut contour_points = Vec::new();
    
    // 画像の外周をスキャンして輪郭を抽出
    // 簡易実装: 矩形の境界を返す（完全な輪郭抽出は複雑なので）
    // TODO: より高度な輪郭抽出アルゴリズム（マーチングスクエアなど）
    
    // とりあえず、アルファ値が閾値以上の領域の矩形境界を返す
    let mut min_x = width;
    let mut max_x = 0;
    let mut min_y = height;
    let mut max_y = 0;
    
    for y in 0..height {
        for x in 0..width {
            let pixel = rgba.get_pixel(x, y);
            if pixel[3] >= alpha_threshold {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }
    
    // 矩形の4隅を輪郭として返す
    if max_x >= min_x && max_y >= min_y {
        contour_points.push((min_x as f32, min_y as f32));
        contour_points.push((max_x as f32, min_y as f32));
        contour_points.push((max_x as f32, max_y as f32));
        contour_points.push((min_x as f32, max_y as f32));
    } else {
        // アルファ値が閾値以上のピクセルがない場合、画像全体を使用
        contour_points.push((0.0, 0.0));
        contour_points.push((width as f32, 0.0));
        contour_points.push((width as f32, height as f32));
        contour_points.push((0.0, height as f32));
    }
    
    contour_points
}

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
    commands.spawn(Camera2d::default());

    // UIテキスト
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

    // イベント用 (非同期MPSC)
    let (tx_cmd, rx_cmd) = tokio::sync::mpsc::channel::<PyCommand>(10);
    // 結果用 (同期Crossbeam - Bevyが受信)
    let (tx_res, rx_res) = bounded::<PyResult>(5);

    // Pythonスレッド起動 (Tokio Runtime内で実行)
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

    // 画像をロード
    let image_path = "texture.png";
    let image_handle = asset_server.load(image_path);

    // TODO: 画像サイズを取得（現在はハードコード）
    let image_width = 512.0;
    let image_height = 512.0;

    commands.insert_resource(ImageData {
        width: image_width,
        height: image_height,
        handle: image_handle.clone(),
    });

    // Create a quad mesh for the deformed image
    let quad_handle = meshes.add(Rectangle::new(image_width, image_height));
    
    // Create initial material (will be updated when deformation happens)
    let material_handle = materials.add(DeformMaterial {
        source_texture: image_handle.clone(),
        inverse_grid_texture: Handle::default(),
        grid_size: Vec2::new(1.0, 1.0),
    });

    // Spawn the deformed image entity using new Bevy 0.15 API
    commands.spawn((
        Mesh2d(quad_handle),
        MeshMaterial2d(material_handle),
        Transform::from_xyz(0.0, 0.0, 0.0),
        DeformedImage,
    ));

    // Pythonへドメイン初期化コマンドを送信
    let epsilon = 100.0;
    println!(
        "Initializing domain: {}x{}, epsilon={}",
        image_width, image_height, epsilon
    );
    let _ = tx_cmd.try_send(PyCommand::InitializeDomain {
        width: image_width,
        height: image_height,
        epsilon,
    });

    // 輪郭線を抽出してPythonに送信
    let full_image_path = format!("assets/{}", image_path);
    let contour = extract_contour_from_image(&full_image_path);
    if !contour.is_empty() {
        println!("Extracted contour with {} points", contour.len());
        let _ = tx_cmd.try_send(PyCommand::SetContour { contour });
    } else {
        println!("No contour extracted, using full image domain");
    }
}

async fn python_thread_loop(
    mut rx_cmd: tokio::sync::mpsc::Receiver<PyCommand>,
    tx_res: Sender<PyResult>,
) {
    pyo3::prepare_freethreaded_python();

    // Pythonオブジェクトをループ外で保持
    let bridge: PyObject = Python::with_gil(|py| {
        let sys = py.import("sys").unwrap();
        let current_dir = env::current_dir().unwrap();
        let script_dir = current_dir.join("scripts");
        if let Ok(path) = sys.getattr("path") {
            if let Ok(path_list) = path.downcast::<PyList>() {
                let _ = path_list.insert(0, script_dir);
            }
        }
        let module = PyModule::import(py, "bevy_bridge").expect("Failed to import bevy_bridge");
        let bridge_class = module.getattr("BevyBridge").expect("No BevyBridge class");
        let bridge_instance = bridge_class.call0().expect("Failed to init BevyBridge");
        bridge_instance.into()
    });

    println!("Rust: Python Thread Ready");

    // Helper function to extract from PyDict
    fn extract_from_dict<T>(dict: &pyo3::Bound<'_, pyo3::types::PyDict>, key: &str) -> Option<T>
    where
        T: for<'a> pyo3::FromPyObject<'a>,
    {
        dict.get_item(key).ok().flatten().and_then(|v| v.extract::<T>().ok())
    }

    loop {
        let Some(cmd) = rx_cmd.recv().await else {
            break;
        };

        Python::with_gil(|py| {
            let bridge_bound = bridge.bind(py);
            match cmd {
                PyCommand::InitializeDomain { width, height, epsilon } => {
                    println!("Rust->Py: InitializeDomain {}x{}, eps={}", width, height, epsilon);
                    if let Err(e) = bridge_bound.call_method1("initialize_domain", (width, height, epsilon)) {
                        eprintln!("Py Error (InitializeDomain): {}", e);
                    } else {
                        let _ = tx_res.send(PyResult::DomainInitialized);
                    }
                }
                PyCommand::SetContour { contour } => {
                    println!("Rust->Py: SetContour with {} points", contour.len());
                    if let Err(e) = bridge_bound.call_method1("set_contour", (contour,)) {
                        eprintln!("Py Error (SetContour): {}", e);
                    }
                }
                PyCommand::AddControlPoint { index, x, y } => {
                    if let Err(e) = bridge_bound.call_method1("add_control_point", (index, x, y)) {
                        eprintln!("Py Error (Add): {}", e);
                    }
                }
                PyCommand::FinalizeSetup => {
                    println!("Rust: Finalizing Setup...");
                    if let Err(e) = bridge_bound.call_method0("finalize_setup") {
                        eprintln!("Py Error (Finalize): {}", e);
                    }
                }
                PyCommand::StartDrag => {
                    if let Err(e) = bridge_bound.call_method0("start_drag_operation") {
                        eprintln!("Py Error (StartDrag): {}", e);
                    }
                }
                PyCommand::UpdatePoint { control_index, x, y } => {
                    // Update the control point position
                    if let Err(e) = bridge_bound.call_method1("update_control_point", (control_index, x, y)) {
                        eprintln!("Py Error (UpdatePoint): {}", e);
                    } else {
                        // Solve and send mapping parameters
                        if let Ok(res) = bridge_bound.call_method0("solve_frame") {
                            if let Ok(params) = res.downcast::<pyo3::types::PyDict>() {
                                let coeffs = extract_from_dict(params, "coefficients");
                                let centers = extract_from_dict(params, "centers");
                                let s = extract_from_dict(params, "s_param");
                                let n = extract_from_dict(params, "n_rbf");
                                let w = extract_from_dict(params, "image_width");
                                let h = extract_from_dict(params, "image_height");
                                let inv_grid = extract_from_dict(params, "inverse_grid");
                                let grid_w = extract_from_dict(params, "grid_width");
                                let grid_h = extract_from_dict(params, "grid_height");
                                
                                if let (Some(coeffs), Some(centers), Some(s), Some(n), Some(w), Some(h), Some(inv_grid), Some(grid_w), Some(grid_h)) = 
                                    (coeffs, centers, s, n, w, h, inv_grid, grid_w, grid_h) {
                                    let _ = tx_res.send(PyResult::MappingParameters {
                                        coefficients: coeffs,
                                        centers,
                                        s_param: s,
                                        n_rbf: n,
                                        image_width: w,
                                        image_height: h,
                                        inverse_grid: inv_grid,
                                        grid_width: grid_w,
                                        grid_height: grid_h,
                                    });
                                }
                            }
                        }
                    }
                }
                PyCommand::EndDrag => {
                    // Call end_drag_operation which may refine the grid
                    if let Err(e) = bridge_bound.call_method0("end_drag_operation") {
                        eprintln!("Py Error (EndDrag): {}", e);
                    } else {
                        // Final solve after drag ends (may use refined grid)
                        if let Ok(res) = bridge_bound.call_method0("solve_frame") {
                            if let Ok(params) = res.downcast::<pyo3::types::PyDict>() {
                                let coeffs = extract_from_dict(params, "coefficients");
                                let centers = extract_from_dict(params, "centers");
                                let s = extract_from_dict(params, "s_param");
                                let n = extract_from_dict(params, "n_rbf");
                                let w = extract_from_dict(params, "image_width");
                                let h = extract_from_dict(params, "image_height");
                                let inv_grid = extract_from_dict(params, "inverse_grid");
                                let grid_w = extract_from_dict(params, "grid_width");
                                let grid_h = extract_from_dict(params, "grid_height");
                                
                                if let (Some(coeffs), Some(centers), Some(s), Some(n), Some(w), Some(h), Some(inv_grid), Some(grid_w), Some(grid_h)) = 
                                    (coeffs, centers, s, n, w, h, inv_grid, grid_w, grid_h) {
                                    let _ = tx_res.send(PyResult::MappingParameters {
                                        coefficients: coeffs,
                                        centers,
                                        s_param: s,
                                        n_rbf: n,
                                        image_width: w,
                                        image_height: h,
                                        inverse_grid: inv_grid,
                                        grid_width: grid_w,
                                        grid_height: grid_h,
                                    });
                                }
                            }
                        }
                    }
                }
                PyCommand::Reset => {
                    if let Err(e) = bridge_bound.call_method0("reset_mesh") {
                        eprintln!("Py Error (Reset): {}", e);
                    }
                }
            }
        });
    }
}

fn receive_python_results(
    channels: Res<PythonChannels>,
    mut mapping_params: ResMut<MappingParameters>,
) {
    while let Ok(res) = channels.rx_result.try_recv() {
        match res {
            PyResult::DomainInitialized => {
                println!("Domain initialized");
            }
            PyResult::MappingParameters {
                coefficients,
                centers,
                s_param,
                n_rbf,
                image_width,
                image_height,
                inverse_grid,
                grid_width,
                grid_height,
            } => {
                println!("Received mapping parameters: {} RBFs, inverse grid {}x{}", n_rbf, grid_width, grid_height);
                mapping_params.coefficients = coefficients;
                mapping_params.centers = centers;
                mapping_params.s_param = s_param;
                mapping_params.n_rbf = n_rbf;
                mapping_params.image_width = image_width;
                mapping_params.image_height = image_height;
                mapping_params.inverse_grid = inverse_grid;
                mapping_params.grid_width = grid_width;
                mapping_params.grid_height = grid_height;
                mapping_params.is_valid = true;
            }
        }
    }
}

fn render_deformed_image(
    mapping_params: Res<MappingParameters>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<DeformMaterial>>,
    query: Query<&MeshMaterial2d<DeformMaterial>, With<DeformedImage>>,
) {
    // マッピングパラメータが有効でない場合は何もしない
    if !mapping_params.is_valid {
        return;
    }

    // Get the material handle
    let Ok(material_2d) = query.get_single() else {
        return;
    };

    // Get the material
    let Some(material) = materials.get_mut(&material_2d.0) else {
        return;
    };

    // Convert inverse grid to texture
    let grid_w = mapping_params.grid_width;
    let grid_h = mapping_params.grid_height;
    
    // Create RG32Float texture for inverse mapping (2 channels for x, y)
    let mut grid_data = Vec::with_capacity(grid_w * grid_h * 8); // 2 floats * 4 bytes each
    
    for y in 0..grid_h {
        for x in 0..grid_w {
            let src_x = mapping_params.inverse_grid[y][x][0];
            let src_y = mapping_params.inverse_grid[y][x][1];
            
            // Write as little-endian f32
            grid_data.extend_from_slice(&src_x.to_le_bytes());
            grid_data.extend_from_slice(&src_y.to_le_bytes());
        }
    }
    
    let inverse_grid_image = Image::new(
        Extent3d {
            width: grid_w as u32,
            height: grid_h as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        grid_data,
        TextureFormat::Rg32Float,
        Default::default(),
    );
    
    // Add to assets and update material
    let inverse_grid_handle = images.add(inverse_grid_image);
    material.inverse_grid_texture = inverse_grid_handle;
    material.grid_size = Vec2::new(grid_w as f32, grid_h as f32);
}

fn draw_control_points(
    control_points: Res<ControlPoints>,
    mut gizmos: Gizmos,
    state: Res<State<AppMode>>,
) {
    // モードによって制御点の色を変える
    let point_color = match state.get() {
        AppMode::Setup => Color::srgb(1.0, 1.0, 0.0),   // YELLOW
        AppMode::Deform => Color::srgb(0.0, 1.0, 1.0),  // CYAN
    };

    for (i, (_, pos)) in control_points.points.iter().enumerate() {
        let color = if Some(i) == control_points.dragging_index {
            Color::srgb(1.0, 0.0, 0.0)  // RED when dragging
        } else {
            point_color
        };
        gizmos.circle_2d(*pos, 8.0, color);
    }
}

fn handle_input(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    channels: Res<PythonChannels>,
    mut control_points: ResMut<ControlPoints>,
    state: Res<State<AppMode>>,
    mut next_state: ResMut<NextState<AppMode>>,
    image_data: Res<ImageData>,
) {
    let Ok(window) = windows.get_single() else { return };
    let Ok((camera, cam_transform)) = camera_q.get_single() else { return };

    // --- 共通: リセット (R) ---
    if keys.just_pressed(KeyCode::KeyR) {
        control_points.points.clear();
        control_points.dragging_index = None;
        let _ = channels.tx_command.try_send(PyCommand::Reset);
        next_state.set(AppMode::Setup);
        return;
    }

    // --- 共通: Enterキーでモード切替 (Setup -> Deform) ---
    if *state.get() == AppMode::Setup && keys.just_pressed(KeyCode::Enter) {
        if control_points.points.is_empty() {
            println!("No control points added!");
            return;
        }
        println!("Switching to Deform Mode");
        let _ = channels.tx_command.try_send(PyCommand::FinalizeSetup);
        next_state.set(AppMode::Deform);
        return;
    }

    // --- マウス操作 ---
    let Some(cursor_pos) = window.cursor_position() else { return };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor_pos) else { return };
    let world_pos = ray.origin.truncate();

    // ワールド座標 -> Python画像座標変換
    // Bevy: center(0,0), Y-Up
    // Python(Image): TopLeft(0,0), Y-Down
    let img_w = image_data.width;
    let img_h = image_data.height;

    let py_x = world_pos.x + img_w / 2.0;
    let py_y = img_h / 2.0 - world_pos.y; // Y軸反転

    // 範囲外チェック
    if py_x < 0.0 || py_x > img_w || py_y < 0.0 || py_y > img_h {
        if control_points.dragging_index.is_some() && buttons.just_released(MouseButton::Left) {
            control_points.dragging_index = None;
            let _ = channels.tx_command.try_send(PyCommand::EndDrag);
        }
        return;
    }

    match state.get() {
        AppMode::Setup => {
            if buttons.just_pressed(MouseButton::Left) {
                // In setup mode, add control point at clicked location
                // Check if we already have a control point nearby
                let threshold = 20.0; // pixels in world space
                let too_close = control_points
                    .points
                    .iter()
                    .any(|(_, pos)| pos.distance(world_pos) < threshold);

                if !too_close {
                    // Add new control point
                    let control_idx = control_points.points.len();
                    control_points.points.push((control_idx, world_pos));

                    // Send to Python
                    let _ = channels.tx_command.try_send(PyCommand::AddControlPoint {
                        index: control_idx,
                        x: py_x,
                        y: py_y,
                    });
                    println!(
                        "Added control point {} at ({:.1}, {:.1})",
                        control_idx, py_x, py_y
                    );
                }
            }
        }
        AppMode::Deform => {
            if buttons.just_pressed(MouseButton::Left) {
                // Find which control point is being clicked
                if let Some(idx) = control_points
                    .points
                    .iter()
                    .position(|(_, pos)| pos.distance(world_pos) < 20.0)
                {
                    control_points.dragging_index = Some(idx);
                    // Notify Python that drag started
                    let _ = channels.tx_command.try_send(PyCommand::StartDrag);
                }
            }

            if buttons.pressed(MouseButton::Left) {
                if let Some(idx) = control_points.dragging_index {
                    if idx < control_points.points.len() {
                        // Update visual position
                        control_points.points[idx].1 = world_pos;

                        // Send update to Python
                        let _ = channels.tx_command.try_send(PyCommand::UpdatePoint {
                            control_index: idx,
                            x: py_x,
                            y: py_y,
                        });
                    }
                }
            }

            if buttons.just_released(MouseButton::Left) {
                if control_points.dragging_index.is_some() {
                    // Notify Python that drag ended (triggers Strategy 2 verification)
                    let _ = channels.tx_command.try_send(PyCommand::EndDrag);
                    control_points.dragging_index = None;
                }
            }
        }
    }
}

fn update_ui_text(
    state: Res<State<AppMode>>,
    mut query: Query<(&mut Text, &mut TextColor), With<ModeText>>,
) {
    if let Ok((mut text, mut color)) = query.get_single_mut() {
        match state.get() {
            AppMode::Setup => {
                **text = "Mode: SETUP\n[Click] Add Point\n[Enter] Start Deform\n[R] Reset".to_string();
                color.0 = Color::srgb(0.0, 1.0, 0.0); // Green
            },
            AppMode::Deform => {
                **text = "Mode: DEFORM\n[Drag] Move Points\n[R] Reset".to_string();
                color.0 = Color::srgb(1.0, 0.5, 0.5); // Redish
            },
        }
    }
}


