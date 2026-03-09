//! GPU deformation path: push pgpm-core coefficients into shader uniforms.
//!
//! Used when the basis is Euclidean Gaussian and the shader can compute
//! basis values directly.

use bevy::prelude::*;
use bevy::sprite::MeshMaterial2d;

use crate::rendering::material::{DeformMaterial, DeformUniform, RBFCenter, RBFCoeff, MAX_RBF_COUNT};
use crate::state::{AlgorithmState, AppState, DeformedImage, ImageInfo};
use crate::rendering::cpu_deform::UseShapeAwareBasis;
use crate::domain::rbf::compute_rbf_scale;

/// System: push pgpm-core coefficients into the GPU material uniform.
pub fn update_deform_material(
    _state: Res<State<AppState>>,
    algo_state: Res<AlgorithmState>,
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
    let Some(ref algo) = algo_state.algorithm else { return };

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

    for (i, src) in algo_state.source_handles.iter().enumerate().take(n_rbf_clamped) {
        params.centers[i] = RBFCenter {
            pos: Vec2::new(src.x as f32, src.y as f32),
            _padding: Vec2::ZERO,
        };
    }

    let s = compute_rbf_scale(
        &algo_state.source_handles,
        image_info.width as f64,
        image_info.height as f64,
    );
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
