//! Deformation rendering: update shader uniforms from pgpm-core coefficients.

use bevy::prelude::*;
use bevy::sprite::MeshMaterial2d;

use crate::rendering::material::{DeformMaterial, DeformUniform, RBFCenter, RBFCoeff, MAX_RBF_COUNT};
use crate::state::{AppState, DeformationState, DeformedImage, ImageInfo};

/// System: push pgpm-core coefficients into the GPU material uniform.
///
/// The coefficient layout from pgpm-core is:
///   c ∈ R^{2×n}, where n = num_rbf_centers + 3 (affine terms)
///   c[(0, i)] = coefficient for u-component of basis i
///   c[(1, i)] = coefficient for v-component of basis i
///   Basis order: [RBF_0, ..., RBF_{n_rbf-1}, const, x, y]
///
/// This exactly matches the shader's coeffs[] array layout.
pub fn update_deform_material(
    _state: Res<State<AppState>>,
    deform_state: Res<DeformationState>,
    image_info: Option<Res<ImageInfo>>,
    mut materials: ResMut<Assets<DeformMaterial>>,
    query: Query<&MeshMaterial2d<DeformMaterial>, With<DeformedImage>>,
) {
    let Some(image_info) = image_info else { return };
    let Some(ref algo) = deform_state.algorithm else { return };

    let Ok(material_2d) = query.get_single() else { return };
    let Some(material) = materials.get_mut(&material_2d.0) else { return };

    let coefficients = algo.coefficients();
    let basis = algo.basis();

    // Get GaussianBasis specifics via the trait
    let n_total = basis.count(); // num_rbf + 3
    let n_rbf = n_total.saturating_sub(3);
    let n_rbf_clamped = n_rbf.min(MAX_RBF_COUNT - 3);

    let mut params = DeformUniform::default();
    params.image_width = image_info.width;
    params.image_height = image_info.height;

    // We need the GaussianBasis centers and scale. Since BasisFunction trait
    // doesn't expose these, we'll get them via downcast. This is the one place
    // where we need concrete type knowledge.
    //
    // For Phase 2, we know it's always GaussianBasis.
    // We extract centers by evaluating identity: the centers are the source_handles.
    // Actually, we can get s and centers from the algorithm's source data.
    // Let's use the deform_state's source_handles as centers, and compute s.

    // Use source handles as RBF centers (they were passed to GaussianBasis::new)
    for (i, src) in deform_state.source_handles.iter().enumerate().take(n_rbf_clamped) {
        params.centers[i] = RBFCenter {
            pos: Vec2::new(src.x as f32, src.y as f32),
            _padding: Vec2::ZERO,
        };
    }

    // Compute s from the source handles (same logic as state.rs)
    let s = compute_s_from_handles(&deform_state.source_handles, image_info.width as f64, image_info.height as f64);
    params.s_param = s as f32;
    params.n_rbf = n_rbf_clamped as u32;

    // Copy coefficients: the first n_rbf + 3 entries
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
