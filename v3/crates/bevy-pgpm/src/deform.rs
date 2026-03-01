//! Deformation rendering: update shader uniforms or mesh positions
//! from pgpm-core coefficients.
//!
//! Two rendering paths:
//! - **GPU path** (Euclidean Gaussian): coefficients are pushed to the shader
//!   uniform, and the vertex shader evaluates f(x) per vertex on the GPU.
//! - **CPU path** (shape-aware Gaussian): f(x) is evaluated on CPU for each
//!   mesh vertex, and mesh POSITION attributes are updated directly.
//!   The shader uniform is set to identity so it becomes a pass-through.

use bevy::prelude::*;
use bevy::sprite::MeshMaterial2d;

use crate::rendering::material::{DeformMaterial, DeformUniform, RBFCenter, RBFCoeff, MAX_RBF_COUNT};
use crate::state::{AppState, DeformationState, DeformedImage, ImageInfo};

/// Resource storing original pixel-coordinate positions of mesh vertices.
/// Needed for CPU displacement: we evaluate f(original_pixel_pos) each frame.
#[derive(Resource)]
pub struct OriginalVertexPositions {
    /// Pixel-space positions (x, y) for each vertex, matching mesh vertex order.
    pub positions: Vec<[f32; 2]>,
}

/// Whether to use CPU-side or GPU-side deformation for the current algorithm.
#[derive(Resource, Default)]
pub struct UseShapeAwareBasis(pub bool);

/// System: push pgpm-core coefficients into the GPU material uniform.
///
/// This is the GPU path — used when the basis is Euclidean Gaussian
/// and the shader can compute basis values directly.
pub fn update_deform_material(
    _state: Res<State<AppState>>,
    deform_state: Res<DeformationState>,
    image_info: Option<Res<ImageInfo>>,
    shape_aware: Option<Res<UseShapeAwareBasis>>,
    mut materials: ResMut<Assets<DeformMaterial>>,
    query: Query<&MeshMaterial2d<DeformMaterial>, With<DeformedImage>>,
) {
    // Skip GPU path when using shape-aware basis (CPU path handles it)
    if shape_aware.map_or(false, |s| s.0) {
        return;
    }

    let Some(image_info) = image_info else { return };
    let Some(ref algo) = deform_state.algorithm else { return };

    let Ok(material_2d) = query.get_single() else { return };
    let Some(material) = materials.get_mut(&material_2d.0) else { return };

    let coefficients = algo.coefficients();
    let basis = algo.basis();

    let n_total = basis.count();
    let n_rbf = n_total.saturating_sub(3);
    let n_rbf_clamped = n_rbf.min(MAX_RBF_COUNT - 3);

    let mut params = DeformUniform::default();
    params.image_width = image_info.width;
    params.image_height = image_info.height;

    for (i, src) in deform_state.source_handles.iter().enumerate().take(n_rbf_clamped) {
        params.centers[i] = RBFCenter {
            pos: Vec2::new(src.x as f32, src.y as f32),
            _padding: Vec2::ZERO,
        };
    }

    let s = compute_s_from_handles(&deform_state.source_handles, image_info.width as f64, image_info.height as f64);
    params.s_param = s as f32;
    params.n_rbf = n_rbf_clamped as u32;

    let n_copy = (n_rbf_clamped + 3).min(MAX_RBF_COUNT).min(n_total);
    for i in 0..n_copy {
        params.coeffs[i] = RBFCoeff {
            x: coefficients[(0, i)] as f32,
            y: coefficients[(1, i)] as f32,
            _padding: Vec2::ZERO,
        };
    }

    material.params = params;
}

/// System: compute deformed positions on CPU and update mesh vertices.
///
/// This is the CPU path — used when the basis is shape-aware Gaussian
/// (geodesic distance) and the shader cannot evaluate basis values.
///
/// The shader uniform is set to identity (n_rbf=0, affine=identity)
/// so the vertex shader is effectively a pass-through.
pub fn cpu_update_mesh_positions(
    _state: Res<State<AppState>>,
    deform_state: Res<DeformationState>,
    image_info: Option<Res<ImageInfo>>,
    shape_aware: Option<Res<UseShapeAwareBasis>>,
    original_verts: Option<Res<OriginalVertexPositions>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<DeformMaterial>>,
    query: Query<(&Mesh2d, &MeshMaterial2d<DeformMaterial>), With<DeformedImage>>,
) {
    // Only run for shape-aware basis
    if !shape_aware.map_or(false, |s| s.0) {
        return;
    }

    let Some(image_info) = image_info else { return };
    let Some(ref algo) = deform_state.algorithm else { return };
    let Some(original_verts) = original_verts else { return };

    let Ok((mesh_handle, material_handle)) = query.get_single() else { return };
    let Some(mesh) = meshes.get_mut(&mesh_handle.0) else { return };

    // Evaluate f(x) at each original vertex position
    let positions: Vec<[f32; 3]> = original_verts.positions.iter().map(|&[px, py]| {
        let pixel_pos = nalgebra::Vector2::new(px as f64, py as f64);
        let deformed = algo.evaluate(pixel_pos);
        // Convert back to world-centered coords
        let wx = deformed.x as f32 - image_info.width * 0.5;
        let wy = image_info.height * 0.5 - deformed.y as f32;
        [wx, wy, 0.0]
    }).collect();

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

    // Set shader to identity pass-through
    if let Some(material) = materials.get_mut(&material_handle.0) {
        let mut params = DeformUniform::default();
        params.image_width = image_info.width;
        params.image_height = image_info.height;
        params.n_rbf = 0;
        // Identity affine: const=(0,0), x=(1,0), y=(0,1)
        params.coeffs[0] = RBFCoeff { x: 0.0, y: 0.0, _padding: Vec2::ZERO };
        params.coeffs[1] = RBFCoeff { x: 1.0, y: 0.0, _padding: Vec2::ZERO };
        params.coeffs[2] = RBFCoeff { x: 0.0, y: 1.0, _padding: Vec2::ZERO };
        material.params = params;
    }
}

fn compute_s_from_handles(handles: &[nalgebra::Vector2<f64>], width: f64, height: f64) -> f64 {
    if handles.len() < 2 {
        return width.max(height) / 4.0;
    }
    let mut total = 0.0;
    for (i, p) in handles.iter().enumerate() {
        let mut min_d = f64::INFINITY;
        for (j, q) in handles.iter().enumerate() {
            if i != j {
                let d = (p - q).norm();
                if d < min_d {
                    min_d = d;
                }
            }
        }
        total += min_d;
    }
    let avg_nn = total / handles.len() as f64;
    avg_nn * 0.8
}
