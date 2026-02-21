use bevy::prelude::*;
use bevy::render::mesh::Indices;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::PrimitiveTopology;

/// Generate a uniform grid mesh for deformation
/// 
/// # Arguments
/// * `size` - The size of the mesh in world units (width, height)
/// * `subdivisions` - Number of subdivisions in each direction (x, y)
/// 
/// # Returns
/// A `Mesh` with positions, UVs, normals, and indices set up for rendering
pub fn create_grid_mesh(size: Vec2, subdivisions: UVec2) -> Mesh {
    let width_segments = subdivisions.x as usize;
    let height_segments = subdivisions.y as usize;
    
    let num_vertices = (width_segments + 1) * (height_segments + 1);
    let mut positions = Vec::with_capacity(num_vertices);
    let mut uvs = Vec::with_capacity(num_vertices);
    let mut normals = Vec::with_capacity(num_vertices);
    
    // Generate vertices
    for y in 0..=height_segments {
        for x in 0..=width_segments {
            let u = x as f32 / width_segments as f32;
            let v = y as f32 / height_segments as f32;
            
            // Position: centered at origin, Y-up (Bevy convention)
            let px = (u - 0.5) * size.x;
            let py = (0.5 - v) * size.y;  // Flip Y to match Bevy's coordinate system
            
            positions.push([px, py, 0.0]);
            uvs.push([u, v]);
            normals.push([0.0, 0.0, 1.0]);
        }
    }
    
    // Generate indices for triangle list
    let mut indices = Vec::with_capacity(width_segments * height_segments * 6);
    for y in 0..height_segments {
        for x in 0..width_segments {
            let base = (y * (width_segments + 1) + x) as u32;
            
            // First triangle
            indices.push(base);
            indices.push(base + (width_segments as u32 + 1));
            indices.push(base + 1);
            
            // Second triangle
            indices.push(base + 1);
            indices.push(base + (width_segments as u32 + 1));
            indices.push(base + (width_segments as u32 + 2));
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
