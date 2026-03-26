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
pub mod mapping;
#[doc(hidden)]
pub mod policy;

// algorithmモジュールは #[doc(hidden)] pub にしている。
// 結合テスト（外部クレートスコープ）から Algorithm<D> に直接アクセスできるようにするため。
// 主要な公開APIは MappingBridge トレイト + ファクトリ関数。
#[doc(hidden)]
pub mod algorithm;

// ─────────────────────────────────────────────
// ファクトリ関数: 主要な公開API
// ─────────────────────────────────────────────

use crate::algorithm::Algorithm;
use crate::basis::BasisFunction;
use crate::mapping::MappingBridge;
use crate::model::domain::Domain;
use crate::model::types::{DomainBounds, MappingParams, SolverConfig};
use crate::policy::{ConformalPolicy, IsometricPolicy};
use nalgebra::Vector2;

/// 等長写像を生成する (D_iso = max{Sigma, 1/sigma})。
///
/// pgpm-core利用者のための主要なエントリーポイント。
/// 具体的な `Algorithm<IsometricPolicy>` 型を隠蔽した
/// `Box<dyn MappingBridge>` を返す。
pub fn create_isometric_mapping(
    basis: Box<dyn BasisFunction>,
    params: MappingParams,
    domain_bounds: DomainBounds,
    source_handles: Vec<Vector2<f64>>,
    grid_resolution: usize,
    fps_k: usize,
    domain: Option<Box<dyn Domain>>,
    solver_config: SolverConfig,
) -> Box<dyn MappingBridge> {
    Box::new(Algorithm::new(
        basis,
        params,
        IsometricPolicy,
        domain_bounds,
        source_handles,
        grid_resolution,
        fps_k,
        domain,
        solver_config,
    ))
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
) -> Box<dyn MappingBridge> {
    Box::new(Algorithm::new(
        basis,
        params,
        ConformalPolicy { delta },
        domain_bounds,
        source_handles,
        grid_resolution,
        fps_k,
        domain,
        solver_config,
    ))
}
