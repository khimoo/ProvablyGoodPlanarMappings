//! # pgpm-core
//!
//! Pure Rust implementation of "Provably Good Planar Mappings" (Poranne & Lipman, 2014)
//!
//! ## Architecture
//!
//! This crate implements Algorithm 1 (Section 5) with a trait-based architecture:
//!
//! - **`ProvablyGoodPlanarMapping` trait**: Defines the complete algorithm with default implementations
//! - **Strategy traits**: Pluggable components (basis function, distortion, regularization, solver)
//! - **`PGPMv2` struct**: Concrete implementation (provides only data access via getters)
//!
//! ## Design Philosophy
//!
//! The trait provides all algorithm logic via default implementations. Concrete types like PGPMv2
//! only need to implement getter methods, achieving complete separation of concerns:
//! - **Data management**: Handled by PGPMv2
//! - **Algorithm logic**: Handled by trait defaults
//! - **Mathematical operations**: Delegated to injected Strategy traits
//!
//! This design ensures Algorithm 1 is implemented once and reused across all concrete types,
//! regardless of their internal data representation or strategy choices.
//!
//! ## Paper References
//!
//! All implementations reference specific equations from:
//! > Poranne, R., & Lipman, Y. (2014). Provably Good Planar Mappings.
//! > ACM Transactions on Graphics (TOG), 33(4), 76.
//!
//! Key equations:
//! - Eq. 3: Mapping evaluation f(x) = Σ c_i φ(||x - x_i||)
//! - Eq. 10-11: Distortion formulas (isometric and conformal)
//! - Eq. 14-17: Active set management strategy
//! - Eq. 18: SOCP formulation
//! - Eq. 21-28: Constraint definitions
//! - Eq. 29-33: Regularization energies
//! - Algorithm 1 (Section 5): Complete mapping algorithm
//!
//! ## Phase 1 Status
//!
//! **Completed**:
//! - ✅ Core trait and default implementations
//! - ✅ PGPMv2 concrete type
//! - ✅ GaussianBasis (Table 1)
//! - ✅ IsometricStrategy (Eq. 10-11)
//! - ✅ BiharmonicRegularization (Eq. 31)
//! - ✅ ClarabelSolver stub (Phase 2 integration)
//!
//! **Phase 2+ (Future)**:
//! - ⬜ Full SOCP solver integration
//! - ⬜ ConformalStrategy (Eq. 12-13)
//! - ⬜ ARAPRegularization (Eq. 32-33)
//! - ⬜ B-Spline and TPS bases (Table 1)
//! - ⬜ Verification strategies (Algorithm 1, step 7)
//!
//! ## Usage Example
//!
//! ```ignore
//! use pgpm_core::*;
//!
//! // Create domain from image contour
//! let domain = DomainInfo {
//!     boundary: DomainBoundary { points: vec![...] },
//!     filling_distance: 0.01,
//! };
//!
//! // Instantiate mapping with strategies
//! let mut mapping = PGPMv2::new(
//!     domain,
//!     Box::new(GaussianBasis::new(1.0)),
//!     Box::new(IsometricStrategy::default()),
//!     Box::new(BiharmonicRegularization::default()),
//!     Box::new(ClarabelSolver),
//! );
//!
//! // Add constraint (handle)
//! mapping.add_handle(0, Vec2::new(0.5, 0.5), Vec2::new(0.6, 0.5))?;
//!
//! // Run algorithm
//! loop {
//!     let result = mapping.algorithm_step()?;
//!     if result.is_converged {
//!         break;
//!     }
//! }
//!
//! // Evaluate mapping at any point
//! let deformed_pos = mapping.evaluate_mapping(Vec2::new(0.3, 0.3));
//!
//! // Verify local injectivity
//! let verification = mapping.verify_local_injectivity()?;
//! ```

// Public modules
pub mod types;
pub mod strategy;
pub mod mapping;
pub mod concrete;
pub mod basis;
pub mod distortion;
pub mod regularization;
pub mod solver;
pub mod bridge;

// Re-export public API
pub use types::*;
pub use strategy::{BasisFunction, DistortionStrategy, RegularizationStrategy, SOCPSolverBackend};
pub use mapping::ProvablyGoodPlanarMapping;
pub use concrete::PGPMv2;
pub use basis::GaussianBasis;
pub use distortion::IsometricStrategy;
pub use regularization::BiharmonicRegularization;
pub use solver::ClarabelSolver;

// Re-export bridge API for bevy-pgpm compatibility
pub use bridge::{
    MappingBridge, PGPMv2Bridge, MappingParams, RegularizationType,
    StepInfo, RefinementResult,
};
