//! Mesh generation for the deformed image.
//!
//! Creates a Delaunay-triangulated mesh from a dense grid of points,
//! optionally clipped to a contour polygon. Each vertex stores its
//! original position as UV for texture lookup.

use bevy::prelude::*;
use bevy::render::mesh::Indices;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::PrimitiveTopology;
use delaunator::{triangulate, Point};
use geo::{Contains, Coord, LineString, Polygon};

/// Generate a contour-clipped mesh suitable for deformation rendering.
///
/// Points are generated on a regular grid within `size`, then filtered
/// to only those inside the contour polygon (if provided). A Delaunay
/// triangulation is computed, and only triangles whose centroids are
/// inside the polygon are kept.
///
/// Vertex attributes:
/// - POSITION: world-space position (centered at origin)
/// - UV_0: normalized texture coordinates [0,1]²
/// - NORMAL: (0, 0, 1)
pub fn create_contour_mesh(
    size: Vec2,
    subdivisions: UVec2,
    contour: &[(f32, f32)],
    holes: &[Vec<(f32, f32)>],
) -> Mesh {
    let width_segments = subdivisions.x as usize;
    let height_segments = subdivisions.y as usize;

    let has_contour = !contour.is_empty();
    let polygon = if has_contour {
        let outer_coords: Vec<Coord<f32>> =
            contour.iter().map(|&(x, y)| Coord { x, y }).collect();
        let hole_rings: Vec<LineString<f32>> = holes
            .iter()
            .map(|hole| {
                let coords: Vec<Coord<f32>> =
                    hole.iter().map(|&(x, y)| Coord { x, y }).collect();
                LineString::from(coords)
            })
            .collect();
        Some(Polygon::new(LineString::from(outer_coords), hole_rings))
    } else {
        None
    };

    let mut valid_points = Vec::new();

    // Add outer contour points (they are on the boundary)
    if has_contour {
        for &(x, y) in contour {
            valid_points.push(Vec2::new(x, y));
        }
    }

    // Add hole contour points (they are on the hole boundaries)
    for hole in holes {
        for &(x, y) in hole {
            valid_points.push(Vec2::new(x, y));
        }
    }

    // Add grid interior points
    for y in 0..=height_segments {
        for x in 0..=width_segments {
            let px = (x as f32 / width_segments as f32) * size.x;
            let py = (y as f32 / height_segments as f32) * size.y;

            if let Some(ref poly) = polygon {
                if poly.contains(&Coord { x: px, y: py }) {
                    valid_points.push(Vec2::new(px, py));
                }
            } else {
                valid_points.push(Vec2::new(px, py));
            }
        }
    }

    // Delaunay triangulation
    let delaunay_points: Vec<Point> = valid_points
        .iter()
        .map(|p| Point {
            x: p.x as f64,
            y: p.y as f64,
        })
        .collect();
    let result = triangulate(&delaunay_points);

    let mut positions = Vec::with_capacity(valid_points.len());
    let mut uvs = Vec::with_capacity(valid_points.len());
    let mut normals = Vec::with_capacity(valid_points.len());
    let mut indices = Vec::new();

    for p in &valid_points {
        let u = p.x / size.x;
        let v = p.y / size.y;
        // Convert to centered world coords
        let bx = p.x - size.x * 0.5;
        let by = size.y * 0.5 - p.y;
        positions.push([bx, by, 0.0]);
        uvs.push([u, v]);
        normals.push([0.0, 0.0, 1.0]);
    }

    for i in (0..result.triangles.len()).step_by(3) {
        let t0 = result.triangles[i];
        let t1 = result.triangles[i + 1];
        let t2 = result.triangles[i + 2];

        let p0 = valid_points[t0];
        let p1 = valid_points[t1];
        let p2 = valid_points[t2];

        let cx = (p0.x + p1.x + p2.x) / 3.0;
        let cy = (p0.y + p1.y + p2.y) / 3.0;

        let keep = if let Some(ref poly) = polygon {
            poly.contains(&Coord { x: cx, y: cy })
        } else {
            true
        };

        if keep {
            indices.push(t0 as u32);
            indices.push(t1 as u32);
            indices.push(t2 as u32);
        }
    }

    // MAIN_WORLD is needed for CPU-side mesh updates (shape-aware basis
    // computes vertex positions on CPU). RENDER_WORLD for GPU rendering.
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}
