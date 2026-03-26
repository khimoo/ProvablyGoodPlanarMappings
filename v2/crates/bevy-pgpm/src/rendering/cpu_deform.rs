//! CPU 変形パス: pgpm-core を介してメッシュ頂点で写像を評価。
//!
//! すべての評価が pgpm-core の `MappingBridge::evaluate_mapping_at()` を
//! 経由するため、任意の基底関数タイプ（ユークリッド Gaussian、
//! Shape-Aware Gaussian 等）で正しく動作する。

use bevy::prelude::*;

use crate::domain::coords::ImageCoords;
use crate::state::{AlgorithmState, DeformedImage, ImageInfo, OriginalVertexPositions};

/// システム: 現在の写像に基づいてメッシュ頂点位置を更新 (Eq. 3)。
///
/// 実行条件: `DeformingSet`（変形中のみ実行）。
///
/// 各アルゴリズムステップの後、元のピクセル空間頂点位置で f(x) を評価し、
/// 変形後のワールド空間位置をメッシュの `POSITION` 属性に書き込む。
pub fn update_cpu_deform(
    algo_state: Res<AlgorithmState>,
    image_info: Option<Res<ImageInfo>>,
    original_positions: Option<Res<OriginalVertexPositions>>,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<&Mesh2d, With<DeformedImage>>,
) {
    let Some(image_info) = image_info else { return };
    let Some(ref algo) = algo_state.algorithm else { return };
    let Some(ref original_positions) = original_positions else { return };
    let Ok(mesh_handle) = query.single() else { return };
    let Some(mesh) = meshes.get_mut(&mesh_handle.0) else { return };

    // 全ての元のピクセル位置で順方向写像 f(x) を評価 (Eq. 3)
    let deformed_pixels = algo.evaluate_mapping_at(&original_positions.pixel_positions);

    // 変形後ピクセル座標 -> ワールド座標に変換してメッシュを更新
    let coords = ImageCoords::new(image_info.width, image_info.height);
    let new_positions: Vec<[f32; 3]> = deformed_pixels
        .iter()
        .map(|p| {
            let world = coords.pixel_to_world(p.x as f32, p.y as f32);
            [world.x, world.y, 0.0]
        })
        .collect();

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, new_positions);
}
