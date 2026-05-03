//! # pgpm-core
//!
//! "Provably Good Planar Mappings" (Poranne & Lipman, 2014) の
//! 純粋な実装。
//!
//! このクレートは論文に記述されたアルゴリズム以外を含めないこと.
//! Bevy、UI、画像処理への依存は一切ない。

pub mod model;
pub mod basis;
pub mod distortion;
pub mod numerics;
pub mod policy;
pub mod algorithm;

// ─────────────────────────────────────────────
// ファクトリ関数: 主要な公開API
// ─────────────────────────────────────────────

use crate::algorithm::Algorithm;
use crate::basis::BasisFunction;
use crate::model::domain::Domain;
use crate::model::types::{DomainBounds, MappingParams, SolverConfig};
use crate::policy::{ConformalPolicy, IsometricPolicy};
use nalgebra::Vector2;

/// 等長写像を生成する (D_iso = max{Sigma, 1/sigma})。
///
/// pgpm-core利用者のための主要なエントリーポイント。
/// `Algorithm` 構造体を直接返す。
pub fn create_isometric_mapping(
    basis: Box<dyn BasisFunction>,
    params: MappingParams,
    domain_bounds: DomainBounds,
    source_handles: Vec<Vector2<f64>>,
    grid_resolution: usize,
    fps_k: usize,
    domain: Option<Box<dyn Domain>>,
    solver_config: SolverConfig,
) -> Algorithm {
    Algorithm::new(
        basis,
        params,
        Box::new(IsometricPolicy),
        domain_bounds,
        source_handles,
        grid_resolution,
        fps_k,
        domain,
        solver_config,
    )
}

/// 等角写像を生成する (D_conf = Sigma / sigma)。
///
/// `delta` は delta > omega(h) を満たす必要がある (Eq. 13)。
pub fn create_conformal_mapping(
    basis: Box<dyn BasisFunction>,
    params: MappingParams,
    delta: f64,
    domain_bounds: DomainBounds,
    source_handles: Vec<Vector2<f64>>,
    grid_resolution: usize,
    fps_k: usize,
    domain: Option<Box<dyn Domain>>,
    solver_config: SolverConfig,
) -> Algorithm {
    Algorithm::new(
        basis,
        params,
        Box::new(ConformalPolicy { delta }),
        domain_bounds,
        source_handles,
        grid_resolution,
        fps_k,
        domain,
        solver_config,
    )
}
