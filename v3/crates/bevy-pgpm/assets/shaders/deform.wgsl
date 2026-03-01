// Deformation vertex shader for Provably Good Planar Mappings.
//
// Evaluates the forward mapping f(x) = Σ c_i φ_i(x) + affine terms
// per vertex. UV coordinates are preserved for texture sampling.
//
// Paper Eq. 3: f(x) = Σ_{i=1}^{n} c_i f_i(x)
// Table 1: f_i(x) = exp(-|x - x_i|² / (2s²)) for Gaussian RBF

#import bevy_sprite::mesh2d_vertex_output::VertexOutput
#import bevy_sprite::mesh2d_functions

struct RBFCenter {
    pos: vec2<f32>,
    _padding: vec2<f32>,
}

struct RBFCoeff {
    x: f32,
    y: f32,
    _padding: vec2<f32>,
}

struct DeformUniform {
    image_width: f32,
    image_height: f32,
    s_param: f32,
    n_rbf: u32,
    centers: array<RBFCenter, 256>,
    coeffs: array<RBFCoeff, 256>,
}

@group(2) @binding(0) var source_texture: texture_2d<f32>;
@group(2) @binding(1) var source_sampler: sampler;
@group(2) @binding(2) var<uniform> uniforms: DeformUniform;

/// Gaussian RBF: φ(r²) = exp(-r² / (2s²))
fn rbf_basis(r_squared: f32, s_param: f32) -> f32 {
    let s2 = s_param * s_param;
    return exp(-r_squared / (2.0 * s2));
}

/// Evaluate forward mapping f(x) = Σ c_i φ_i(x) + affine terms
/// Basis layout: [RBF_0, ..., RBF_{n-1}, const, x, y]
/// Coeffs layout matches: coeffs[0..n_rbf] = RBF, coeffs[n_rbf..n_rbf+3] = affine
fn evaluate_forward_mapping(pos: vec2<f32>) -> vec2<f32> {
    var result = vec2<f32>(0.0);
    let n_rbf = uniforms.n_rbf;

    // RBF contributions
    for (var i: u32 = 0u; i < n_rbf; i = i + 1u) {
        let center = uniforms.centers[i].pos;
        let delta = pos - center;
        let r_squared = dot(delta, delta);
        let phi = rbf_basis(r_squared, uniforms.s_param);

        result.x += uniforms.coeffs[i].x * phi;
        result.y += uniforms.coeffs[i].y * phi;
    }

    // Affine terms: f_{n-2}=1, f_{n-1}=x, f_n=y
    let affine_idx = n_rbf;
    result.x += uniforms.coeffs[affine_idx].x
              + uniforms.coeffs[affine_idx + 1u].x * pos.x
              + uniforms.coeffs[affine_idx + 2u].x * pos.y;
    result.y += uniforms.coeffs[affine_idx].y
              + uniforms.coeffs[affine_idx + 1u].y * pos.x
              + uniforms.coeffs[affine_idx + 2u].y * pos.y;

    return result;
}

@vertex
fn vertex(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    // Convert from world-centered coords to domain (pixel) coords
    let pixel_x = position.x + uniforms.image_width * 0.5;
    let pixel_y = uniforms.image_height * 0.5 - position.y;
    let pixel_pos = vec2<f32>(pixel_x, pixel_y);

    // Apply the deformation mapping
    let deformed_pixel = evaluate_forward_mapping(pixel_pos);

    // Convert back to world-centered coords
    let deformed_x = deformed_pixel.x - uniforms.image_width * 0.5;
    let deformed_y = uniforms.image_height * 0.5 - deformed_pixel.y;

    let local_pos = vec4<f32>(deformed_x, deformed_y, position.z, 1.0);

    let model = mesh2d_functions::get_world_from_local(instance_index);
    out.world_position = mesh2d_functions::mesh2d_position_local_to_world(model, local_pos);
    out.position = mesh2d_functions::mesh2d_position_local_to_clip(model, local_pos);
    out.uv = uv;

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(source_texture, source_sampler, in.uv);
    return color;
}
