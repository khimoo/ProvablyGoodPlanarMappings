use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::state::{DeformedImage, MappingParameters};
use super::material::DeformMaterial;

pub fn render_deformed_image(
    mapping_params: Res<MappingParameters>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<DeformMaterial>>,
    query: Query<&MeshMaterial2d<DeformMaterial>, With<DeformedImage>>,
) {
    if !mapping_params.is_valid {
        return;
    }

    let Ok(material_2d) = query.get_single() else {
        return;
    };

    let Some(material) = materials.get_mut(&material_2d.0) else {
        return;
    };

    let grid_w = mapping_params.grid_width;
    let grid_h = mapping_params.grid_height;

    let mut grid_data = Vec::with_capacity(grid_w * grid_h * 8);

    for y in 0..grid_h {
        for x in 0..grid_w {
            let src_x = mapping_params.inverse_grid[y][x][0];
            let src_y = mapping_params.inverse_grid[y][x][1];

            grid_data.extend_from_slice(&src_x.to_le_bytes());
            grid_data.extend_from_slice(&src_y.to_le_bytes());
        }
    }

    let inverse_grid_image = Image::new(
        Extent3d {
            width: grid_w as u32,
            height: grid_h as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        grid_data,
        TextureFormat::Rg32Float,
        Default::default(),
    );

    let inverse_grid_handle = images.add(inverse_grid_image);
    material.inverse_grid_texture = inverse_grid_handle;
    material.grid_size = Vec2::new(grid_w as f32, grid_h as f32);
}
