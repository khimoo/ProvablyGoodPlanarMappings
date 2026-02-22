use bevy::prelude::*;
use crate::state::DebugVisualization;

/// シンプルなデバッグ視覚化システム
/// Dキーで表示/非表示を切り替え
pub fn toggle_debug_viz(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut debug_viz: ResMut<DebugVisualization>,
) {
    if keyboard.just_pressed(KeyCode::KeyD) {
        debug_viz.show = !debug_viz.show;
        println!("Debug visualization: {}", if debug_viz.show { "ON" } else { "OFF" });
    }
}

/// デバッグ情報を描画
pub fn draw_debug_viz(
    mut gizmos: Gizmos,
    debug_viz: Res<DebugVisualization>,
) {
    if !debug_viz.show {
        return;
    }

    // 輪郭線を描画（緑）
    if debug_viz.contour.len() > 1 {
        for i in 0..debug_viz.contour.len() {
            let p1 = debug_viz.contour[i];
            let p2 = debug_viz.contour[(i + 1) % debug_viz.contour.len()];
            gizmos.line_2d(p1, p2, Color::srgb(0.0, 1.0, 0.0));
        }
    }

    // コロケーション点を描画（青、小さい点）
    for point in &debug_viz.collocation_points {
        gizmos.circle_2d(*point, 1.0, Color::srgba(0.3, 0.5, 1.0, 0.3));
    }

    // Active Set を描画（赤、大きい点）
    for point in &debug_viz.active_set {
        gizmos.circle_2d(*point, 3.0, Color::srgb(1.0, 0.0, 0.0));
    }
}
