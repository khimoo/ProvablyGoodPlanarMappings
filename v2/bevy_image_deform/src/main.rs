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
use std::time::Duration; // 追加
use tokio::sync::watch; // 追加

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
    LoadImage { path: String, epsilon: f32 },
    AddControlPoint { index: usize, x: f32, y: f32 },
    FinalizeSetup, // セットアップ完了、ソルバー構築指示
    Reset,
}

// --- 通信結果 ---
enum PyResult {
    MeshInit {
        vertices: Vec<f32>,
        indices: Vec<u32>,
        uvs: Vec<f32>,
        image_width: f32,
        image_height: f32,
    },
    DeformUpdate(Vec<f32>),
}

// --- リソース ---
#[derive(Resource)]
struct PythonChannels {
    tx_command: tokio::sync::mpsc::Sender<PyCommand>,
    rx_result: Receiver<PyResult>,
    tx_coords: watch::Sender<Option<(usize, f32, f32)>>,
}

#[derive(Resource, Default)]
struct ControlPoints {
    // (頂点インデックス, 表示座標)
    points: Vec<(usize, Vec2)>,
    dragging_index: Option<usize>,
}#[derive(Resource, Default)]
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

    // イベント用 (非同期MPSC)
    let (tx_cmd, rx_cmd) = tokio::sync::mpsc::channel::<PyCommand>(10);
    // 結果用 (同期Crossbeam - Bevyが受信)
    let (tx_res, rx_res) = bounded::<PyResult>(5); 
    // 座標同期用ウォッチチャンネル (初期値: None)
    let (tx_coords, rx_coords) = watch::channel::<Option<(usize, f32, f32)>>(None);

    // Pythonスレッド起動 (Tokio Runtime内で実行)
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(python_thread_loop(rx_cmd, tx_res, rx_coords));
    });

    commands.insert_resource(PythonChannels {
        tx_command: tx_cmd.clone(),
        rx_result: rx_res,
        tx_coords,
    });

    // メッシュ
    let image_path = "assets/texture.png";
    let epsilon = 100.0;
    println!("Requesting mesh gen for: {}", image_path);

    // Pythonへ初期化コマンドを送信
    let _ = tx_cmd.try_send(PyCommand::LoadImage { 
        path: image_path.to_string(), 
        epsilon 
    });
}

fn update_mesh_and_gizmos(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
    mut query: Query<(Entity, &mut Mesh2d), With<DeformableMesh>>,
    channels: Res<PythonChannels>,
    control_points: Res<ControlPoints>,
    mut gizmos: Gizmos,
    state: Res<State<AppMode>>,
    mut mesh_data: ResMut<MeshData>,
) {
    // 1. メッシュ更新
    while let Ok(res) = channels.rx_result.try_recv() {
        match res {
            PyResult::MeshInit { vertices, indices, uvs, image_width, image_height } => {
                println!("Got MeshInit: {} verts, {}x{}", vertices.len()/2, image_width, image_height);
                
                // Bevy's coordinate system
                // Vertices from Python are in Image Coords (0..W, 0..H), Y-Down.
                // We need to convert to World Coords for Bevy display (Center Origin, Y-Up).
                let world_positions: Vec<Vec2> = vertices
                    .chunks(2)
                    .map(|v| Vec2::new(
                        v[0] - image_width / 2.0, 
                        image_height / 2.0 - v[1]
                    ))
                    .collect();

                // Build Mesh
                let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
                let positions_3d: Vec<[f32; 3]> = world_positions.iter().map(|p| [p.x, p.y, 0.0]).collect();
                mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions_3d);
                
                // UVs
                let uv_vecs: Vec<[f32; 2]> = uvs.chunks(2).map(|v| [v[0], v[1]]).collect();
                mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv_vecs);
                
                mesh.insert_indices(Indices::U32(indices));

                let mesh_handle = meshes.add(mesh);

                // Check if we already have the entity
                if let Ok((entity, mut m2d)) = query.get_single_mut() {
                    m2d.0 = mesh_handle;
                } else {
                    // Spawn new
                    commands.spawn((
                        Sprite::from_image(asset_server.load("texture.png")),
                        Transform::from_xyz(0.0, 0.0, -1.0),
                    ));

                    commands.spawn((
                        Mesh2d(mesh_handle),
                        MeshMaterial2d(materials.add(ColorMaterial {
                            color: Color::WHITE.with_alpha(0.1),
                            ..default()
                        })),
                        Transform::from_xyz(0.0, 0.0, 0.0),
                        DeformableMesh,
                    ));
                }

                // Update MeshData
                mesh_data.vertex_count = vertices.len() / 2;
                mesh_data.initial_positions = world_positions;
                mesh_data.image_width = image_width;
                mesh_data.image_height = image_height;
            },
            PyResult::DeformUpdate(flat_verts) => {
               if let Ok((_, mesh_handle)) = query.get_single() {
                   if let Some(mesh) = meshes.get_mut(&mesh_handle.0) {
                       let img_w = mesh_data.image_width;
                       let img_h = mesh_data.image_height;
                       // Python returns Image Coords. Convert to World.
                       let positions: Vec<[f32; 3]> = flat_verts
                        .chunks(2)
                        .map(|v| [
                            v[0] - img_w / 2.0,      
                            -(v[1] - img_h / 2.0), // Y-Flip relative to center
                            0.0
                        ])
                        .collect();
                       mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
                   }
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
    for (_, mesh_2d) in query.iter() {
        if let Some(mesh) = meshes.get(&mesh_2d.0) {
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
async fn python_thread_loop(
    mut rx_cmd: tokio::sync::mpsc::Receiver<PyCommand>,
    tx_res: Sender<PyResult>,
    mut rx_coords: watch::Receiver<Option<(usize, f32, f32)>>
) {
    pyo3::prepare_freethreaded_python();
    
    // Pythonオブジェクトをループ外で保持するために PyObject (Py<PyAny>) を取得
    let bridge: PyObject = Python::with_gil(|py| {
        let sys = py.import("sys").unwrap();
        let current_dir = env::current_dir().unwrap();
        let script_dir = current_dir.join("scripts");
        if let Ok(path) = sys.getattr("path") {
            if let Ok(path_list) = path.downcast::<PyList>() {
                let _ = path_list.insert(0, script_dir);
            }
        }
        let module = PyModule::import(py, "deform_algo").expect("Failed to import deform_algo");
        let bridge_class = module.getattr("BevyBridge").expect("No BevyBridge class");
        // Rustのスレッド内で保持するため、GILに束縛されない PyObject に変換しておく
        let bridge_instance = bridge_class.call0().expect("Failed to init BevyBridge");
        bridge_instance.into()
    });

    println!("Rust: Python Thread Waiting for Mesh Data...");

    loop {
        let mut should_solve = false;
        
        // select! でイベントを待つ
        tokio::select! {
            Some(cmd) = rx_cmd.recv() => {
                Python::with_gil(|py| {
                    let bridge_bound = bridge.bind(py);
                    match cmd {
                        PyCommand::LoadImage { path, epsilon } => {
                             println!("Rust->Py: LoadImage {}, eps={}", path, epsilon);
                             match bridge_bound.call_method1("load_image_and_generate_mesh", (path, epsilon)) {
                                Ok(res_tuple) => {
                                    if let Ok((verts, indices, uvs, w, h)) = res_tuple.extract::<(Vec<f32>, Vec<u32>, Vec<f32>, f32, f32)>() {
                                        let _ = tx_res.send(PyResult::MeshInit {
                                            vertices: verts,
                                            indices,
                                            uvs,
                                            image_width: w,
                                            image_height: h,
                                        });
                                    } else {
                                        eprintln!("Py Error: Failed to extract mesh data tuple");
                                    }
                                },
                                Err(e) => eprintln!("Py Error (LoadImage): {}", e),
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
                            } else {
                                should_solve = true;
                            }
                        }
                        PyCommand::Reset => {
                             if let Err(e) = bridge_bound.call_method0("reset_mesh") {
                                eprintln!("Py Error (Reset): {}", e);
                            } else {
                                should_solve = true;
                            }
                        }
                    }
                });
            }
            Ok(_) = rx_coords.changed() => {
                let val = *rx_coords.borrow_and_update();
                if let Some((index, x, y)) = val {
                    Python::with_gil(|py| {
                        let bridge_bound = bridge.bind(py);
                         // println!("Rust->Py: Update Point {} -> ({:.1}, {:.1})", index, x, y);
                         if let Err(e) = bridge_bound.call_method1("update_control_point", (index, x, y)) {
                            eprintln!("Py Error (Update): {}", e);
                        } else {
                            should_solve = true;
                        }
                    });
                }
            }
        }

        if should_solve {
            let res_verts: Option<Vec<f32>> = Python::with_gil(|py| {
                let bridge_bound = bridge.bind(py);
                match bridge_bound.call_method0("solve_frame") {
                    Ok(res) => {
                         res.extract::<Vec<f32>>().ok()
                    },
                    Err(e) => {
                        eprintln!("Py Solve Error: {}", e);
                        None
                    }
                }
            });

            if let Some(vertices) = res_verts {
                if !vertices.is_empty() {
                    let _ = tx_res.send(PyResult::DeformUpdate(vertices));
                }
            }
        }
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
    mesh_data: Res<MeshData>,
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

                        // Setupモードの追加コマンドは MPSC で確実に送る
                        // try_send ではなく blocking send でもいいが、Async Channel なので blocking_send も使える
                        // ここではイベント頻度が低いので try_send でOK、ただし容量に注意
                        let _ = channels.tx_command.try_send(PyCommand::AddControlPoint {
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
                        // 修正: 座標更新は Watch Channel で送る
                        // これで「常に最新の値」だけが保持される
                        let _ = channels.tx_coords.send(Some((idx, py_x, py_y)));
                        // println!("Sent watch update: idx={}, ({:.1}, {:.1})", idx, py_x, py_y);
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


