use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology},
        render_asset::RenderAssetUsages,
    },
};
use crossbeam_channel::{bounded, Receiver, Sender};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};
use std::env;

// --- 定数 ---
const IMAGE_WIDTH: f32 = 512.0;
const IMAGE_HEIGHT: f32 = 512.0;
const GRID_ROWS: usize = 15;
const GRID_COLS: usize = 15;
const EXPECTED_VERTICES: usize = GRID_ROWS * GRID_COLS;

// --- アプリケーションの状態 (モード) ---
#[derive(States, Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
enum AppMode {
    #[default]
    Setup,   // 点を追加するモード (変形なし)
    Deform,  // 点をドラッグして変形するモード
}

// --- 通信コマンド ---
enum PyCommand {
    AddControlPoint { index: usize, x: f32, y: f32 },
    FinalizeSetup, // セットアップ完了、ソルバー構築指示
    UpdateControlPoint { index: usize, x: f32, y: f32 },
    Reset,
}

// --- リソース ---
#[derive(Resource)]
struct PythonChannels {
    tx_command: Sender<PyCommand>,
    rx_result: Receiver<Vec<f32>>,
}

#[derive(Resource, Default)]
struct ControlPoints {
    // (頂点インデックス, 表示座標)
    points: Vec<(usize, Vec2)>,
    dragging_index: Option<usize>,
}

#[derive(Component)]
struct DeformableMesh;

#[derive(Component)]
struct ModeText;

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
        .init_state::<AppMode>()
        .init_resource::<ControlPoints>()
        .add_systems(Startup, setup)
        .add_systems(Update, (
            handle_input,
            update_mesh_and_gizmos,
            update_ui_text,
        ))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
) {
    commands.spawn(Camera2d::default());

    // UIテキスト
    commands.spawn((
        Text::new("Mode: Setup\n[Click] Add Point\n[Enter] Start Deform\n[R] Reset"),
        TextColor(Color::WHITE), // 初期色
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        ModeText,
    ));

    // 通信チャンネル
    let (tx_cmd, rx_cmd) = bounded::<PyCommand>(1);
    let (tx_res, rx_res) = bounded::<Vec<f32>>(5);

    // Pythonスレッド起動
    std::thread::spawn(move || {
        python_thread_loop(rx_cmd, tx_res);
    });

    commands.insert_resource(PythonChannels {
        tx_command: tx_cmd,
        rx_result: rx_res,
    });

    // メッシュ
    let mesh = create_grid_mesh(GRID_ROWS, GRID_COLS, IMAGE_WIDTH, IMAGE_HEIGHT);
    let mesh_handle = meshes.add(mesh);
    let texture_handle = asset_server.load("texture.png");

    commands.spawn((
        Mesh2d(mesh_handle),
        MeshMaterial2d(materials.add(ColorMaterial {
            texture: Some(texture_handle),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        DeformableMesh,
    ));
}

fn python_thread_loop(rx_cmd: Receiver<PyCommand>, tx_res: Sender<Vec<f32>>) {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let sys = py.import("sys").unwrap();
        let current_dir = env::current_dir().unwrap();
        let script_dir = current_dir.join("scripts");
        if let Ok(path) = sys.getattr("path") {
            if let Ok(path_list) = path.downcast::<PyList>() {
                let _ = path_list.insert(0, script_dir);
            }
        }
        let module = match PyModule::import(py, "deform_algo") {
            Ok(m) => m,
            Err(e) => { eprintln!("Rust Import Error: {}", e); return; }
        };

        let bridge_class = module.getattr("BevyBridge").expect("No BevyBridge class");
        let bridge = bridge_class.call0().expect("Failed to init BevyBridge");

        // 初期化
        let _ = bridge.call_method1("initialize_mesh", (IMAGE_WIDTH, IMAGE_HEIGHT, GRID_ROWS, GRID_COLS));
        println!("Rust: Python Ready.");

        loop {
            // コマンド受信 (ブロッキング)
            if let Ok(cmd) = rx_cmd.recv() {
                let mut should_solve = false;
                match cmd {
                    PyCommand::AddControlPoint { index, x, y } => {
                        // Setupモード: 追加のみ、計算なし
                        if let Err(e) = bridge.call_method1("add_control_point", (index, x, y)) {
                            eprintln!("Py Error (Add): {}", e);
                        }
                    }
                    PyCommand::FinalizeSetup => {
                        // Setup完了: ソルバー構築
                        println!("Rust: Finalizing Setup...");
                        if let Err(e) = bridge.call_method0("finalize_setup") {
                            eprintln!("Py Error (Finalize): {}", e);
                        } else {
                            should_solve = true; // 初回の変形なし形状を取得するために一度solve
                        }
                    }
                    PyCommand::UpdateControlPoint { index, x, y } => {
                        // Deformモード: 更新して計算
                        if let Err(e) = bridge.call_method1("update_control_point", (index, x, y)) {
                            eprintln!("Py Error (Update): {}", e);
                        } else {
                            should_solve = true;
                        }
                    }
                    PyCommand::Reset => {
                        if let Err(e) = bridge.call_method0("reset_mesh") {
                            eprintln!("Py Error (Reset): {}", e);
                        } else {
                            should_solve = true; // 初期形状に戻す
                        }
                    }
                }

                if should_solve {
                    match bridge.call_method0("solve_frame") {
                        Ok(res) => {
                            if let Ok(vertices) = res.extract::<Vec<f32>>() {
                                if !vertices.is_empty() {
                                    let _ = tx_res.send(vertices);
                                }
                            }
                        },
                        Err(e) => eprintln!("Py Solve Error: {}", e),
                    }
                }
            }
        }
    });
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
) {
    let Ok(window) = windows.get_single() else { return };
    let Ok((camera, cam_transform)) = camera_q.get_single() else { return };

    // --- 共通: リセット (R) ---
    if keys.just_pressed(KeyCode::KeyR) {
        control_points.points.clear();
        control_points.dragging_index = None;
        let _ = channels.tx_command.send(PyCommand::Reset);
        next_state.set(AppMode::Setup); // Setupモードに戻る
        println!("Reset -> Mode: Setup");
        return;
    }

    // --- 共通: Enterキーでモード切替 (Setup -> Deform) ---
    if *state.get() == AppMode::Setup && keys.just_pressed(KeyCode::Enter) {
        if control_points.points.len() < 3 {
            println!("Warning: Add at least 3 points for stable deformation.");
            // 続行してもよいが、警告
        }
        let _ = channels.tx_command.send(PyCommand::FinalizeSetup);
        next_state.set(AppMode::Deform);
        println!("Mode: Setup -> Deform");
        return;
    }

    // --- マウス操作 ---
    let Some(cursor_pos) = window.cursor_position() else { return };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor_pos) else { return };
    let world_pos = ray.origin.truncate();
    let py_x = world_pos.x + IMAGE_WIDTH / 2.0;
    let py_y = world_pos.y + IMAGE_HEIGHT / 2.0;

    // 範囲外チェック
    if py_x < 0.0 || py_x > IMAGE_WIDTH || py_y < 0.0 || py_y > IMAGE_HEIGHT {
        if control_points.dragging_index.is_some() && buttons.just_released(MouseButton::Left) {
            control_points.dragging_index = None;
        }
        return;
    }

    match state.get() {
        AppMode::Setup => {
            // --- Setupモード: クリックで追加のみ ---
            if buttons.just_pressed(MouseButton::Left) {
                // 重複チェック
                let hit = control_points.points.iter().any(|(_, pos)| pos.distance(world_pos) < 15.0);
                if !hit {
                    // グリッド座標計算
                    let c = ((py_x / IMAGE_WIDTH) * (GRID_COLS - 1) as f32).round() as usize;
                    let r = ((py_y / IMAGE_HEIGHT) * (GRID_ROWS - 1) as f32).round() as usize;
                    let c = c.clamp(0, GRID_COLS - 1);
                    let r = r.clamp(0, GRID_ROWS - 1);
                    let v_idx = r * GRID_COLS + c;

                    if !control_points.points.iter().any(|(pid, _)| *pid == v_idx) {
                        control_points.points.push((v_idx, world_pos));
                        // 追加コマンドだけ送る
                        let _ = channels.tx_command.send(PyCommand::AddControlPoint {
                            index: v_idx, x: py_x, y: py_y
                        });
                        println!("Added Point (Setup): {}", v_idx);
                    }
                }
            }
        },
        AppMode::Deform => {
            // --- Deformモード: ドラッグで移動 ---
            if buttons.just_pressed(MouseButton::Left) {
                // ヒット判定
                if let Some(idx) = control_points.points.iter().position(|(_, pos)| pos.distance(world_pos) < 20.0) {
                    control_points.dragging_index = Some(idx);
                }
            }

            if buttons.pressed(MouseButton::Left) {
                if let Some(idx) = control_points.dragging_index {
                    if idx < control_points.points.len() {
                        control_points.points[idx].1 = world_pos;
                        // 更新コマンドを送る (try_sendで間引き)
                        let cmd = PyCommand::UpdateControlPoint { index: idx, x: py_x, y: py_y };
                        match channels.tx_command.try_send(cmd) {
                            Ok(_) => {},
                            Err(crossbeam_channel::TrySendError::Full(_)) => {}, // スキップ
                            Err(_) => {},
                        }
                    }
                }
            }

            if buttons.just_released(MouseButton::Left) {
                control_points.dragging_index = None;
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

fn update_mesh_and_gizmos(
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<&Mesh2d, With<DeformableMesh>>,
    channels: Res<PythonChannels>,
    control_points: Res<ControlPoints>,
    mut gizmos: Gizmos,
    state: Res<State<AppMode>>,
) {
    // 1. メッシュ更新
    if let Ok(flat) = channels.rx_result.try_recv() {
        if flat.len() == EXPECTED_VERTICES * 2 {
            for mesh_handle in query.iter() {
                if let Some(mesh) = meshes.get_mut(mesh_handle.id()) {
                    let positions: Vec<[f32; 3]> = flat
                        .chunks(2)
                        .map(|v| [v[0] - IMAGE_WIDTH/2.0, v[1] - IMAGE_HEIGHT/2.0, 0.0])
                        .collect();
                    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                }
            }
        }
    }

    // 2. 制御点とワイヤーフレーム描画
    // モードによって制御点の色を変える
    let point_color = match state.get() {
        AppMode::Setup => Color::srgb(1.0, 1.0, 0.0), // YELLOW
        AppMode::Deform => Color::srgb(0.0, 1.0, 1.0), // CYAN
    };

    for (i, (_, pos)) in control_points.points.iter().enumerate() {
        let color = if Some(i) == control_points.dragging_index { Color::srgb(1.0, 0.0, 0.0) } else { point_color }; // RED
        gizmos.circle_2d(*pos, 8.0, color);
    }

    // ワイヤーフレーム
    for mesh_handle in query.iter() {
        if let Some(mesh) = meshes.get(mesh_handle.id()) {
            if let (Some(pos), Some(Indices::U32(inds))) = (
                mesh.attribute(Mesh::ATTRIBUTE_POSITION).and_then(|a| a.as_float3()),
                mesh.indices(),
            ) {
                let max_idx = inds.iter().map(|i| *i as usize).max().unwrap_or(0);
                if pos.len() <= max_idx { continue; }

                for i in (0..inds.len()).step_by(3) {
                    let a = Vec3::from(pos[inds[i] as usize]).truncate();
                    let b = Vec3::from(pos[inds[i+1] as usize]).truncate();
                    let c = Vec3::from(pos[inds[i+2] as usize]).truncate();
                    gizmos.line_2d(a, b, Color::WHITE.with_alpha(0.2));
                    gizmos.line_2d(b, c, Color::WHITE.with_alpha(0.2));
                    gizmos.line_2d(c, a, Color::WHITE.with_alpha(0.2));
                }
            }
        }
    }
}

fn create_grid_mesh(rows: usize, cols: usize, width: f32, height: f32) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    let mut pos = Vec::new();
    let mut uvs = Vec::new();
    let mut inds = Vec::new();

    for r in 0..rows {
        for c in 0..cols {
            let u = c as f32 / (cols - 1) as f32;
            let v = r as f32 / (rows - 1) as f32;
            pos.push([(u - 0.5) * width, (v - 0.5) * height, 0.0]);
            uvs.push([u, 1.0 - v]);
        }
    }

    for r in 0..(rows - 1) {
        for c in 0..(cols - 1) {
            let i = r * cols + c;
            inds.extend_from_slice(&[
                i as u32, (i + 1) as u32, (i + cols + 1) as u32,
                i as u32, (i + cols + 1) as u32, (i + cols) as u32
            ]);
        }
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(inds));
    mesh
}
