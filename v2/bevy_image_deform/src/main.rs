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
use image::GenericImageView;
use delaunator::{Point, triangulate};

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
    InitializeMesh { vertices: Vec<f32>, indices: Vec<u32> },
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

#[derive(Resource, Default)]
struct MeshData {
    // 初期頂点位置 (World座標)
    initial_positions: Vec<Vec2>,
    vertex_count: usize,
    // 画像サイズを保持
    image_width: f32,
    image_height: f32,
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
        .init_resource::<MeshData>()
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
    mut mesh_data: ResMut<MeshData>,
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
        tx_command: tx_cmd.clone(),
        rx_result: rx_res,
    });

    // メッシュ
    // 画像読み込みとメッシュ生成
    let image_path = "assets/texture.png";
    println!("Generating mesh from: {}", image_path);
    let sample_stride = 15;
    let (mesh, flat_verts, indices, initial_positions, img_w, img_h) = create_mesh_from_image_alpha(image_path, sample_stride);

    // リソースに保存
    mesh_data.vertex_count = flat_verts.len() / 2;
    mesh_data.initial_positions = initial_positions;
    mesh_data.image_width = img_w;
    mesh_data.image_height = img_h;

    // Pythonへ初期メッシュデータを送信
    let _ = tx_cmd.send(PyCommand::InitializeMesh { 
        vertices: flat_verts, 
        indices: indices 
    });

    let mesh_handle = meshes.add(mesh);
    // let texture_handle = asset_server.load("texture.png"); // メッシュにはテクスチャを貼らない

    // 1. 背景画像の表示 (Sprite)
    commands.spawn((
        Sprite::from_image(asset_server.load("texture.png")),
        Transform::from_xyz(0.0, 0.0, -1.0), // メッシュより奥に
    ));

    // 2. メッシュの表示 (ワイヤーフレーム用、テクスチャなし)
    commands.spawn((
        Mesh2d(mesh_handle),
        MeshMaterial2d(materials.add(ColorMaterial {
            color: Color::WHITE.with_alpha(0.1), // 半透明の白
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        DeformableMesh,
    ));
}fn python_thread_loop(rx_cmd: Receiver<PyCommand>, tx_res: Sender<Vec<f32>>) {
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

        println!("Rust: Python Thread Waiting for Mesh Data...");

        loop {
            // コマンド受信 (ブロッキング)
            if let Ok(cmd) = rx_cmd.recv() {
                let mut should_solve = false;
                match cmd {
                    PyCommand::InitializeMesh { vertices, indices } => {
                        println!("Rust->Py: Init Mesh ({} verts, {} tris)", vertices.len()/2, indices.len()/3);
                        // Vec<f32> -> List[List[float, float]]
                        let py_verts = PyList::empty(py);
                        for chunk in vertices.chunks(2) {
                            let _ = py_verts.append(PyList::new(py, chunk).unwrap());
                        }
                        
                        // indices はそのままタプルとして渡せるか、リストにするか
                        // BevyBridge側で list を受け取るならこれでOK
                        if let Err(e) = bridge.call_method1("initialize_mesh_from_data", (py_verts, indices)) {
                             eprintln!("Py Error (Init): {}", e);
                        }
                    }
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
    mesh_data: Res<MeshData>,
) {
    let Ok(window) = windows.get_single() else { return };
    let Ok((camera, cam_transform)) = camera_q.get_single() else { return };

    // --- 共通: リセット (R) ---
    if keys.just_pressed(KeyCode::KeyR) {
        control_points.points.clear();
        control_points.dragging_index = None;
        let _ = channels.tx_command.send(PyCommand::Reset);
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
        let _ = channels.tx_command.send(PyCommand::FinalizeSetup);
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
    // 元の変換式: py_x = world_x + W/2, py_y = H/2 - world_y
    // ここでメッシュデータに保存された実際の画像サイズを使用する
    let img_w = mesh_data.image_width;
    let img_h = mesh_data.image_height;

    let py_x = world_pos.x + img_w / 2.0;
    let py_y = img_h / 2.0 - world_pos.y; // Y軸反転

    // 範囲外チェック
    if py_x < 0.0 || py_x > img_w || py_y < 0.0 || py_y > img_h {
        if control_points.dragging_index.is_some() && buttons.just_released(MouseButton::Left) {
            control_points.dragging_index = None;
        }
        return;
    }

    match state.get() {
        AppMode::Setup => {
            if buttons.just_pressed(MouseButton::Left) {
                // 近傍点探索
                let threshold = 15.0; // px
                let mut closest_idx = None;
                let mut min_dist = threshold;

                for (i, pos) in mesh_data.initial_positions.iter().enumerate() {
                    let d = pos.distance(world_pos);
                    if d < min_dist {
                        min_dist = d;
                        closest_idx = Some(i);
                    }
                }

                if let Some(v_idx) = closest_idx {
                    if !control_points.points.iter().any(|(pid, _)| *pid == v_idx) {
                        let snap_pos = mesh_data.initial_positions[v_idx];
                        control_points.points.push((v_idx, snap_pos));
                        
                        let snap_py_x = snap_pos.x + img_w / 2.0;
                        let snap_py_y = img_h / 2.0 - snap_pos.y; // Y軸反転

                        let _ = channels.tx_command.send(PyCommand::AddControlPoint {
                            index: v_idx, x: snap_py_x, y: snap_py_y
                        });
                        println!("Added Point: {} ({:.1}, {:.1})", v_idx, snap_py_x, snap_py_y);
                    }
                }
            }
        },
        AppMode::Deform => {
            if buttons.just_pressed(MouseButton::Left) {
                if let Some(idx) = control_points.points.iter().position(|(_, pos)| pos.distance(world_pos) < 20.0) {
                    control_points.dragging_index = Some(idx);
                }
            }

            if buttons.pressed(MouseButton::Left) {
                 if let Some(idx) = control_points.dragging_index {
                    if idx < control_points.points.len() {
                        control_points.points[idx].1 = world_pos;
                        let _ = channels.tx_command.try_send(PyCommand::UpdateControlPoint { 
                            index: idx, x: py_x, y: py_y 
                        });
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
    mesh_data: Res<MeshData>,
) {
    // 1. メッシュ更新
    if let Ok(flat) = channels.rx_result.try_recv() {
        if flat.len() == mesh_data.vertex_count * 2 {
            let img_w = mesh_data.image_width;
            let img_h = mesh_data.image_height;

            for mesh_handle in query.iter() {
                if let Some(mesh) = meshes.get_mut(mesh_handle.id()) {
                    let positions: Vec<[f32; 3]> = flat
                        .chunks(2)
                        .map(|v| [
                            v[0] - img_w / 2.0,      // ImageX(0..W) -> WorldX(-W/2..W/2)
                            -(v[1] - img_h / 2.0), // ImageY(0..H) -> WorldY(H/2..-H/2) (Y軸反転)
                            0.0
                        ])
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

// 修正された関数: 画像アルファ値に基づくメッシュ生成（凹部フィルタリング付き）
// 戻り値: (BevyMesh, FlatCoords(Python用), Indices, InitialVec2(World用), img_w, img_h)
fn create_mesh_from_image_alpha(image_path: &str, stride: u32) -> (Mesh, Vec<f32>, Vec<u32>, Vec<Vec2>, f32, f32) {
    let img = image::open(image_path).expect("Failed to open image");
    let (w, h) = img.dimensions();
    let width_f = w as f32;
    let height_f = h as f32;

    let mut points: Vec<Point> = Vec::new();
    let mut uvs = Vec::new();
    let mut flat_verts = Vec::new();
    let mut world_points_vec2 = Vec::new();

    // 1. 点群サンプリング
    for y in (0..h).step_by(stride as usize) {
        for x in (0..w).step_by(stride as usize) {
            let pixel = img.get_pixel(x, y);
            if pixel[3] > 10 { // アルファ閾値
                // Delaunatorはf64要求
                points.push(Point { x: x as f64, y: y as f64 });
                
                // UV (Y反転なし/ありは画像の仕様次第。ここでは標準的なUV)
                uvs.push([x as f32 / width_f, 1.0 - (y as f32 / height_f)]);

                // Python用 (画像座標系)
                flat_verts.push(x as f32);
                flat_verts.push(y as f32);

                // Bevy World用 (中心基準)
                // 画像座標 (0,0)=TopLeft -> Bevy (0,0)=Center
                // y=0 -> +H/2, y=H -> -H/2
                // w_x = x - W/2
                // w_y = -(y - H/2) = H/2 - y
                world_points_vec2.push(Vec2::new(
                    x as f32 - width_f / 2.0,
                    height_f / 2.0 - y as f32
                ));
            }
        }
    }

    // 2. 三角形分割
    let triangulation = triangulate(&points);
    let mut filtered_indices = Vec::new();

    // 3. 三角形フィルタリング (凹部削除)
    for i in (0..triangulation.triangles.len()).step_by(3) {
        let i0 = triangulation.triangles[i];
        let i1 = triangulation.triangles[i+1];
        let i2 = triangulation.triangles[i+2];

        let p0 = &points[i0];
        let p1 = &points[i1];
        let p2 = &points[i2];

        // 重心計算
        let cx = (p0.x + p1.x + p2.x) / 3.0;
        let cy = (p0.y + p1.y + p2.y) / 3.0;

        // 画像範囲チェック
        if cx >= 0.0 && cx < w as f64 && cy >= 0.0 && cy < h as f64 {
            // 重心のピクセルが透明なら、その三角形は除外する
            let alpha = img.get_pixel(cx as u32, cy as u32)[3];
            if alpha > 10 {
                filtered_indices.push(i0 as u32);
                filtered_indices.push(i1 as u32);
                filtered_indices.push(i2 as u32);
            }
        }
    }

    // 4. メッシュ構築
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    let positions: Vec<[f32; 3]> = world_points_vec2.iter().map(|p| [p.x, p.y, 0.0]).collect();

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(filtered_indices.clone()));

    (mesh, flat_verts, filtered_indices, world_points_vec2, width_f, height_f)
}fn create_grid_mesh(rows: usize, cols: usize, width: f32, height: f32) -> Mesh {
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
