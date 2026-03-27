//! bevy-pgpm: 証明付き良好な平面写像のインタラクティブUI。
//!
//! Phase 2 実装: Gaussian 基底 + 等長歪みのみ。
//! SOCP ベースの変形アルゴリズムに pgpm-core を使用し、
//! レンダリング/入力/UI に Bevy を使用。
//!
//! 使用方法:
//!   cargo run -p bevy-pgpm
//!
//! assets/ ディレクトリに texture.png を配置する。
//! 全ての操作は画面上のコントロールパネルから:
//! 1. セットアップモード: クリックしてハンドルを配置、次に「Start Deforming」をクリック。
//! 2. 変形モード: ハンドルをドラッグして画像を変形。
//! 3. K 上限、正則化タイプ、lambda をパネルボタンで調整。

use bevy::{
    prelude::*,
    camera::{ClearColorConfig, Viewport},
    window::WindowResized,
};

use log::info;

use bevy_pgpm::{
    input::handle_input,
    lifecycle::{load_image, update_deformation},
    rendering::update_cpu_deform,
    state::*,
    ui,
};

/// このクレート自身の assets/ ディレクトリの絶対パスを返す。
///
/// 用途:
/// - Bevy の `AssetPlugin` ルート（シェーダ、フォント）
/// - デフォルト画像の絶対パス解決
fn crate_asset_dir() -> String {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set (run via `cargo run`)");
    let dir = std::path::PathBuf::from(&manifest).join("assets");
    assert!(
        dir.is_dir(),
        "Assets directory not found: {}\n\
         Expected crates/bevy-pgpm/assets/ to exist.",
        dir.display(),
    );
    dir.to_string_lossy().into_owned()
}

/// デフォルトテクスチャ画像の絶対パスを解決。
fn default_image_path(asset_dir: &str) -> String {
    let p = std::path::PathBuf::from(asset_dir).join("texture.png");
    assert!(
        p.exists(),
        "Default image not found: {}\n\
         Place a texture.png in the crates/bevy-pgpm/assets/ directory.",
        p.display(),
    );
    p.to_string_lossy().into_owned()
}

/// 変形モードでのみ実行されるシステム。
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
struct DeformingSet;

fn main() {
    let asset_dir = crate_asset_dir();
    let default_image = default_image_path(&asset_dir);
    info!("Asset root: {}", asset_dir);

    App::new()
        .add_plugins(DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "PGPM - Provably Good Planar Mappings".into(),
                    resolution: bevy::window::WindowResolution::new(1280, 800),
                    ..default()
                }),
                ..default()
            })
            .set(AssetPlugin {
                file_path: asset_dir,
                ..default()
            })
        )
        .init_state::<AppState>()
        .init_resource::<AlgorithmState>()
        .init_resource::<DragState>()
        .init_resource::<DeformationInfo>()
        .init_resource::<AlgoParams>()
        .insert_resource(ImagePathConfig::new(default_image))
        .configure_sets(Update, DeformingSet.run_if(in_state(AppState::Deforming)))
        .add_systems(Startup, (setup_camera, ui::spawn_control_panel))
        .add_systems(Update, (
            load_image,
            setup_camera_scale,
            handle_input,
        ))
        .add_systems(Update, (
            update_deformation
                .in_set(DeformingSet)
                .before(update_cpu_deform),
            update_cpu_deform
                .in_set(DeformingSet),
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
            ui::on_export,
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

/// スタートアップ: シーンカメラとUIカメラを生成。
///
/// ウィンドウを分割するために2つのカメラを使用:
///   - シーンカメラ (MainCamera): ビューポートを左側に制限、
///     デフォルトレンダーレイヤー0で Mesh2d エンティティをレンダリング。
///   - UIカメラ: ウィンドウ全体をカバー、レンダーレイヤー1（シーン
///     エンティティなし）で IsDefaultUiCamera 経由で UI オーバーレイのみをレンダリング。
fn setup_camera(mut commands: Commands) {
    // シーンカメラ: デフォルトレンダーレイヤー0、ビューポートは setup_camera_viewport で設定。
    commands.spawn((Camera2d::default(), MainCamera));

    // UIカメラ: レンダーレイヤー1（空）で Mesh2d の二重レンダリングを防止。
    // IsDefaultUiCamera は全てのUIノードをこのカメラに向ける。
    commands.spawn((
        Camera2d::default(),
        Camera {
            order: 1,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        bevy::camera::visibility::RenderLayers::layer(1),
        IsDefaultUiCamera,
    ));
}

/// システム: シーンカメラのビューポートをUIパネルの左側エリアに設定し、
/// 投影を画像にフィットするようにスケール。ImageInfo の変更または
/// ウィンドウリサイズ時に再実行。
fn setup_camera_scale(
    image_info: Option<Res<ImageInfo>>,
    windows: Query<&Window>,
    mut camera_q: Query<(&mut Camera, &mut Projection), With<MainCamera>>,
    mut resize_events: MessageReader<WindowResized>,
) {
    let Some(image_info) = image_info else { return };

    let resized = resize_events.read().last().is_some();
    if !image_info.is_changed() && !resized {
        return;
    }

    let Ok(window) = windows.single() else { return };
    let Ok((mut camera, mut projection)) = camera_q.single_mut() else { return };

    // 物理ピクセルでビューポートを計算（Bevy の Viewport に必要）。
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

    // 投影をスケールして画像をビューポート内にマージン付きでフィット。
    let logical_w = viewport_w as f32 / scale_factor;
    let logical_h = physical_h as f32 / scale_factor;

    let margin = 0.9;
    let scale_x = (logical_w * margin) / image_info.width;
    let scale_y = (logical_h * margin) / image_info.height;
    let scale = scale_x.min(scale_y);

    // Projection enum を通じて正射影のスケールを更新。
    if let Projection::Orthographic(ref mut ortho) = *projection {
        ortho.scale = 1.0 / scale;

        info!(
            "Camera viewport: {}x{} physical, image: {}x{}, scale: {}",
            viewport_w, physical_h, image_info.width, image_info.height, ortho.scale
        );
    }
}
