//! ライフサイクルシステム: 画像の読み込み/再読み込みとアルゴリズムステップ。

use bevy::prelude::*;
use bevy::image::Image as BevyImage;
use bevy::mesh::VertexAttributeValues;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use log::{info, warn, error};
use image::GenericImageView;

use crate::domain::coords::ImageCoords;
use crate::domain::image_loader::extract_contour_from_image;
use crate::rendering::create_contour_mesh;
use crate::state::{
    AlgorithmState, AppState, DeformationInfo, DeformedImage, ImageInfo, ImagePathConfig,
    OriginalVertexPositions,
};

/// システム: `ImagePathConfig.needs_reload` が設定されたときに画像を読み込む（または再読み込み）。
pub fn load_image(
    mut commands: Commands,
    mut path_config: ResMut<ImagePathConfig>,
    mut images: ResMut<Assets<BevyImage>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    existing_image: Query<Entity, With<DeformedImage>>,
    mut algo_state: ResMut<AlgorithmState>,
    mut deform_info: ResMut<DeformationInfo>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    if !path_config.needs_reload {
        return;
    }
    path_config.needs_reload = false;

    let abs_path = &path_config.abs_path;
    info!("Loading image from: {}", abs_path);

    // `image` クレートで画像を読み込む（寸法、輪郭、および GPU テクスチャ用）。
    // 手動で読み込んで Assets<Image> に挿入することで、AssetServer の
    // パス解決問題を完全に回避。
    let img = match image::open(abs_path) {
        Ok(img) => img,
        Err(e) => {
            error!("Failed to load image '{}': {}", abs_path, e);
            return;
        }
    };

    let (w, h) = img.dimensions();
    let image_width = w as f32;
    let image_height = h as f32;
    info!("Image dimensions: {}x{}", w, h);

    let contour_data = extract_contour_from_image(abs_path);
    let contour = contour_data.outer;
    let holes = contour_data.holes;

    if contour.is_empty() {
        info!("No contour extracted, using full image domain");
    } else {
        info!(
            "Extracted contour with {} points, {} hole(s)",
            contour.len(),
            holes.len(),
        );
    }

    // RGBA8 に変換して Bevy の Assets<Image> に直接挿入。
    // AssetServer のパス解決問題を完全に回避。
    let rgba = img.to_rgba8();
    let bevy_image = BevyImage::new(
        Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        TextureDimension::D2,
        rgba.into_raw(),
        TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    );
    let image_handle = images.add(bevy_image);

    commands.insert_resource(ImageInfo {
        width: image_width,
        height: image_height,
        handle: image_handle.clone(),
        contour: contour.clone(),
        holes: holes.clone(),
    });

    // 再読み込みの場合は以前の画像エンティティを削除
    for entity in existing_image.iter() {
        commands.entity(entity).despawn();
    }

    // 画像変更時に状態をリセット
    algo_state.reset();
    *deform_info = DeformationInfo::default();
    next_state.set(AppState::Setup);

    // メッシュを作成（ドロネー三角形分割）
    let grid_mesh = create_contour_mesh(
        Vec2::new(image_width, image_height),
        UVec2::new(200, 200),
        &contour,
        &holes,
    );

    let mesh_handle = meshes.add(grid_mesh);

    // CPU 変形パス用に元の頂点位置を抽出
    let world_positions = meshes
        .get(&mesh_handle)
        .and_then(|m| m.attribute(Mesh::ATTRIBUTE_POSITION))
        .and_then(|attr| match attr {
            VertexAttributeValues::Float32x3(v) => Some(v.clone()),
            _ => None,
        })
        .unwrap_or_default();

    let coords = ImageCoords::new(image_width, image_height);
    let pixel_positions: Vec<nalgebra::Vector2<f64>> = world_positions
        .iter()
        .map(|[wx, wy, _]| {
            let (px, py) = coords.world_to_pixel(Vec2::new(*wx, *wy));
            nalgebra::Vector2::new(px as f64, py as f64)
        })
        .collect();

    commands.insert_resource(OriginalVertexPositions {
        pixel_positions,
        world_positions,
    });

    let material_handle = materials.add(ColorMaterial {
        texture: Some(image_handle),
        ..default()
    });

    commands.spawn((
        Mesh2d(mesh_handle),
        MeshMaterial2d(material_handle),
        Transform::from_xyz(0.0, 0.0, 0.0),
        DeformedImage,
    ));
}

/// システム: 必要に応じてアルゴリズムを1ステップ実行。
///
/// 実行条件: DeformingSet 経由で `in_state(AppState::Deforming)`。
pub fn update_deformation(
    mut algo_state: ResMut<AlgorithmState>,
    mut deform_info: ResMut<DeformationInfo>,
) {
    // ドラッグ中またはアルゴリズムが収束していない間は反復を継続。
    // 収束は pgpm-core で判定（Algorithm 1: max_distortion ≤ K
    // かつアクティブ集合が安定）。
    let needs_more = algo_state.needs_solve
        || (deform_info.step_count > 0 && !deform_info.converged);

    if !needs_more {
        return;
    }

    let targets: Vec<nalgebra::Vector2<f64>> = algo_state.target_handles.clone();

    // フレームごとに Algorithm 1 を正確に1ステップ実行（SOCP はブロッキング）。
    if let Some(ref mut algo) = algo_state.algorithm {
        match algo.step(&targets) {
            Ok(step_info) => {
                deform_info.max_distortion = step_info.max_distortion;
                deform_info.active_set_size = step_info.active_set_size;
                deform_info.stable_set_size = step_info.stable_set_size;
                deform_info.converged = step_info.converged;
                deform_info.step_count += 1;
            }
            Err(e) => {
                warn!("SOCP solve failed: {:?}", e);
            }
        }
    }

    algo_state.needs_solve = false;
}
