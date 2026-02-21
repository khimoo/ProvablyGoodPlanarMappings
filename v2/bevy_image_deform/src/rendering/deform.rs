use bevy::prelude::*;

use crate::state::{DeformedImage, MappingParameters};
use super::material::{DeformMaterial, DeformUniform, RBFCenter, RBFCoeff, MAX_RBF_COUNT};

pub fn render_deformed_image(
    mapping_params: Res<MappingParameters>,
    mut materials: ResMut<Assets<DeformMaterial>>,
    query: Query<&MeshMaterial2d<DeformMaterial>, With<DeformedImage>>,
) {
    if !mapping_params.is_valid || mapping_params.n_rbf == 0 {
        return;
    }

    let Ok(material_2d) = query.get_single() else {
        return;
    };

    let Some(material) = materials.get_mut(&material_2d.0) else {
        return;
    };

    let n_rbf = mapping_params.n_rbf.min(MAX_RBF_COUNT);

    // Build uniform data
    let mut params = DeformUniform::default();
    params.image_width = mapping_params.image_width;
    params.image_height = mapping_params.image_height;
    params.s_param = mapping_params.s_param;
    params.n_rbf = n_rbf as u32;

    // Fill centers
    for (i, center) in mapping_params.centers.iter().enumerate().take(n_rbf) {
        params.centers[i] = RBFCenter {
            pos: Vec2::new(center[0], center[1]),
            _padding: Vec2::ZERO,
        };
    }

    // Fill coefficients
    let coeffs_x = mapping_params.coefficients.get(0).cloned().unwrap_or_default();
    let coeffs_y = mapping_params.coefficients.get(1).cloned().unwrap_or_default();

    for i in 0..(n_rbf + 3).min(MAX_RBF_COUNT) {
        params.coeffs[i] = RBFCoeff {
            x: coeffs_x.get(i).copied().unwrap_or(0.0),
            y: coeffs_y.get(i).copied().unwrap_or(0.0),
            _padding: Vec2::ZERO,
        };
    }

    material.params = params;
}
