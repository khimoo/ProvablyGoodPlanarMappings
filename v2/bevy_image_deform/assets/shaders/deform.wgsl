#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@group(2) @binding(0) var source_texture: texture_2d<f32>;
@group(2) @binding(1) var source_sampler: sampler;
@group(2) @binding(2) var inverse_grid_texture: texture_2d<f32>;
@group(2) @binding(3) var inverse_grid_sampler: sampler;

struct DeformUniform {
    grid_size: vec2<f32>,
}

@group(2) @binding(4) var<uniform> deform_uniform: DeformUniform;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // in.uv is the output pixel position in [0, 1] range
    // We need to find the source pixel position using the inverse mapping
    
    // Sample the inverse grid texture to get source coordinates
    // The inverse grid stores (src_x, src_y) for each output position
    let inverse_coords = textureSample(inverse_grid_texture, inverse_grid_sampler, in.uv);
    
    // Get image dimensions
    let image_size = vec2<f32>(textureDimensions(source_texture));
    
    // Convert source coordinates from pixel space to UV space [0, 1]
    let src_uv = inverse_coords.rg / image_size;
    
    // Sample the source texture at the inverse-mapped position
    let color = textureSample(source_texture, source_sampler, src_uv);
    
    return color;
}
