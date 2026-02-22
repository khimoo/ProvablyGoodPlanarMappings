use bevy::prelude::*;
use bevy::render::mesh::Indices;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::PrimitiveTopology;
use delaunator::{Point, triangulate};
use geo::{Contains, Coord, LineString, Polygon};

/// Generate a contour-clipped mesh for deformation
pub fn create_contour_mesh(size: Vec2, subdivisions: UVec2, contour: &[(f32, f32)]) -> Mesh {
    let width_segments = subdivisions.x as usize;
    let height_segments = subdivisions.y as usize;

    // 1. geoクレートのPolygonを作成（内外判定用）
    let coords: Vec<Coord<f32>> = contour.iter().map(|&(x, y)| Coord { x, y }).collect();
    let polygon = Polygon::new(LineString::from(coords), vec![]);

    // 2. メッシュの頂点を集める（輪郭の点 ＋ 内部のグリッド点）
    let mut valid_points = Vec::new();

    // エッジを綺麗に保つため、輪郭の頂点を追加
    for &(x, y) in contour {
        valid_points.push(Vec2::new(x, y));
    }

    // 滑らかに変形させるため、シルエット内部のグリッド点を追加
    for y in 0..=height_segments {
        for x in 0..=width_segments {
            let px = (x as f32 / width_segments as f32) * size.x;
            let py = (y as f32 / height_segments as f32) * size.y;
            let pt = Coord { x: px, y: py };

            // シルエットの内側にある点だけを採用
            if polygon.contains(&pt) {
                valid_points.push(Vec2::new(px, py));
            }
        }
    }

    // 3. Delaunay（ドロネー）分割で三角形のネットワークを生成
    let delaunay_points: Vec<Point> = valid_points
        .iter()
        .map(|p| Point { x: p.x as f64, y: p.y as f64 })
        .collect();
    let result = triangulate(&delaunay_points);

    // 4. Bevy用の頂点バッファを構築
    let mut positions = Vec::with_capacity(valid_points.len());
    let mut uvs = Vec::with_capacity(valid_points.len());
    let mut normals = Vec::with_capacity(valid_points.len());
    let mut indices = Vec::new();

    for p in &valid_points {
        let u = p.x / size.x;
        let v = p.y / size.y;

        // Bevy座標系（中央原点、Y-up）への変換
        let bx = p.x - size.x * 0.5;
        let by = size.y * 0.5 - p.y;

        positions.push([bx, by, 0.0]);
        uvs.push([u, v]);
        normals.push([0.0, 0.0, 1.0]);
    }

    // 5. 不要な三角形（シルエット外のへこんだ部分）をカットする
    for i in (0..result.triangles.len()).step_by(3) {
        let t0 = result.triangles[i];
        let t1 = result.triangles[i + 1];
        let t2 = result.triangles[i + 2];

        let p0 = valid_points[t0];
        let p1 = valid_points[t1];
        let p2 = valid_points[t2];

        // 三角形の中心点（重心）を計算
        let cx = (p0.x + p1.x + p2.x) / 3.0;
        let cy = (p0.y + p1.y + p2.y) / 3.0;

        // 重心がシルエットの内側にある三角形だけをインデックスに追加
        if polygon.contains(&Coord { x: cx, y: cy }) {
            indices.push(t0 as u32);
            indices.push(t1 as u32);
            indices.push(t2 as u32);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}
