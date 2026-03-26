//! レンダリングモジュール: メッシュ生成と CPU 変形。

pub mod mesh;
pub mod cpu_deform;

pub use mesh::create_contour_mesh;
pub use cpu_deform::update_cpu_deform;
