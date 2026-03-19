//! Core data types for the algorithm
//!
//! This module defines all fundamental data structures used throughout pgpm-core.
//! Type aliases are provided for common nalgebra types.

use std::fmt;

/// 2D vector type (f64)
pub type Vec2 = nalgebra::Vector2<f64>;
/// 2x2 matrix type (f64)
pub type Mat2 = nalgebra::Matrix2<f64>;

/// Handle identifier (constraint point)
pub type HandleId = usize;

/// Result type with AlgorithmError
pub type Result<T> = std::result::Result<T, AlgorithmError>;

/// Algorithm errors
#[derive(Clone, Debug)]
pub enum AlgorithmError {
    /// SOCP solver failed
    SolverFailed(String),
    /// Invalid domain configuration
    InvalidDomain(String),
    /// Invalid handle configuration
    InvalidHandle(String),
    /// Numerical error
    NumericalError(String),
    /// Geometry error
    GeometryError(String),
}

impl fmt::Display for AlgorithmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlgorithmError::SolverFailed(msg) => write!(f, "Solver failed: {}", msg),
            AlgorithmError::InvalidDomain(msg) => write!(f, "Invalid domain: {}", msg),
            AlgorithmError::InvalidHandle(msg) => write!(f, "Invalid handle: {}", msg),
            AlgorithmError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            AlgorithmError::GeometryError(msg) => write!(f, "Geometry error: {}", msg),
        }
    }
}

impl std::error::Error for AlgorithmError {}

/// Handle information (constraint point)
#[derive(Clone, Debug)]
pub struct HandleInfo {
    pub id: HandleId,
    pub position: Vec2,
    pub target_value: Vec2,
    pub radius: f64,
}

/// Domain boundary representation (simplified for Phase 1)
#[derive(Clone, Debug)]
pub struct DomainBoundary {
    /// Boundary points (in order)
    pub points: Vec<Vec2>,
}

impl DomainBoundary {
    /// Check if point is inside domain (simple convex check)
    pub fn contains(&self, point: Vec2) -> bool {
        // Phase 1: Simplified check for rectangular domains
        // In Phase 2+, use proper point-in-polygon
        self.points.iter().all(|p| {
            // Very basic check: assume axis-aligned rectangle
            point.x >= p.x.min(self.points[1].x) && point.x <= p.x.max(self.points[1].x)
                && point.y >= p.y.min(self.points[1].y) && point.y <= p.y.max(self.points[1].y)
        })
    }

    /// Distance to nearest boundary point
    pub fn distance_to_boundary(&self, point: Vec2) -> f64 {
        self.points
            .iter()
            .map(|p| (point - p).norm())
            .fold(f64::INFINITY, f64::min)
    }
}

/// Domain information
#[derive(Clone, Debug)]
pub struct DomainInfo {
    pub boundary: DomainBoundary,
    /// Eq. 5: Filling distance η
    pub filling_distance: f64,
}

/// Active set information
#[derive(Clone, Debug)]
pub struct ActiveSetInfo {
    pub centers: Vec<Vec2>,
    pub is_active: Vec<bool>,
    /// σ(x_i) values at each center
    pub sigma_values: Vec<f64>,
}

impl Default for ActiveSetInfo {
    fn default() -> Self {
        Self {
            centers: vec![],
            is_active: vec![],
            sigma_values: vec![],
        }
    }
}

/// Algorithm state tracking
#[derive(Clone, Debug)]
pub struct AlgorithmState {
    pub step_counter: usize,
    pub is_converged: bool,
    pub current_distortion: f64,
}

impl Default for AlgorithmState {
    fn default() -> Self {
        Self {
            step_counter: 0,
            is_converged: false,
            current_distortion: f64::INFINITY,
        }
    }
}

/// Distortion information
#[derive(Clone, Debug)]
pub struct DistortionInfo {
    /// Maximum distortion value at any point
    pub max_distortion: f64,
    /// Points with high distortion (σ > K_high)
    pub local_maxima: Vec<(Vec2, f64)>,
}

impl Default for DistortionInfo {
    fn default() -> Self {
        Self {
            max_distortion: 1.0,
            local_maxima: vec![],
        }
    }
}

/// Result of a single algorithm step
#[derive(Clone, Debug)]
pub struct AlgorithmStepResult {
    pub step_num: usize,
    pub distortion_info: DistortionInfo,
    pub is_converged: bool,
}

/// Verification result
#[derive(Clone, Debug, PartialEq)]
pub enum VerificationResult {
    /// Local injectivity confirmed (det J > 0 everywhere)
    LocallyInjective,
    /// Fold-overs detected (det J ≤ 0)
    HasFoldOvers(Vec<Vec2>),
    /// Cannot guarantee (Phase 3 future work)
    CannotGuarantee,
}

/// Cone constraint for SOCP
#[derive(Clone, Debug)]
pub struct ConeConstraint {
    /// Center point in domain
    pub center: Vec2,
    /// Basis gradients
    pub basis_grads: Vec<Vec2>,
    /// Upper bound for constraint
    pub bound: f64,
}

/// Energy term representation
#[derive(Clone, Debug)]
pub enum EnergyTerms {
    /// Quadratic form: c^T Q c + q^T c
    Quadratic {
        matrix: Vec<Vec<f64>>,
        linear: Vec<f64>,
    },
    /// Handle constraints: f(x_handle) = target
    Handles(Vec<HandleConstraint>),
}

/// Single handle constraint
#[derive(Clone, Debug)]
pub struct HandleConstraint {
    pub position: Vec2,
    pub target: Vec2,
    pub weight: f64,
}

/// SOCP problem formulation (Eq. 18)
#[derive(Clone, Debug)]
pub struct SOCPProblem {
    pub constraints: Vec<ConeConstraint>,
    pub energy: EnergyTerms,
    pub handle_constraints: Vec<HandleConstraint>,
}
