//! レンダリングモジュール: メッシュ生成、CPU 変形、画像エクスポート。

pub mod mesh;
pub mod cpu_deform;
pub mod export;

pub use mesh::create_contour_mesh;
pub use cpu_deform::update_cpu_deform;
