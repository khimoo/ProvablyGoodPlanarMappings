//! Common type definitions for the PGPM algorithm.
//!
//! All types map directly to paper notation:
//! - Eq. 3: coefficient matrix c ∈ R^{2×n}
//! - Section 3: distortion types
//! - Algorithm 1: algorithm state

use nalgebra::{DMatrix, Vector2};

// ───────────────────────────────────────────────────────────────
// Eq. 3: Coefficient matrix c ∈ R^{2×n}
// c = [c_1, c_2, ..., c_n], c_i = (c¹_i, c²_i)^T
// We store this as a DMatrix<f64> of shape (2, n).
// Row 0 → c¹ coefficients (u component)
// Row 1 → c² coefficients (v component)
// ───────────────────────────────────────────────────────────────
pub type CoefficientMatrix = DMatrix<f64>;

/// Section 3 "Distortion": type of distortion measure.
#[derive(Debug, Clone)]
pub enum DistortionType {
    /// D_iso(x) = max{Σ(x), 1/σ(x)}
    Isometric,
    /// D_conf(x) = Σ(x) / σ(x)
    /// δ must satisfy δ > ω(h) (Eq. 13)
    Conformal { delta: f64 },
}

/// Eq. 5: Bounding box of domain Ω.
#[derive(Debug, Clone)]
pub struct DomainBounds {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

/// Algorithm 1 input parameters.
#[derive(Debug, Clone)]
pub struct AlgorithmParams {
    /// Distortion type and upper bound K (Eq. 4: D(z_j) ≤ K)
    pub distortion_type: DistortionType,
    pub k_bound: f64,

    /// Regularization weight λ (Eq. 1, 18)
    pub lambda_reg: f64,

    /// Regularization type and mixing weights
    pub regularization: RegularizationType,
}

/// Section 5.4: regularization energy types.
#[derive(Debug, Clone)]
pub enum RegularizationType {
    /// E_bh only (Eq. 31)
    Biharmonic,
    /// E_arap only (Eq. 33)
    Arap,
    /// E_pos + λ_bh * E_bh + λ_arap * E_arap
    /// Paper Section 6: Figure 5 uses E_pos + 10^{-2} E_arap,
    ///                   Figure 8 uses E_pos + 10^{-1} E_bh
    Mixed { lambda_bh: f64, lambda_arap: f64 },
}

/// Algorithm 1 internal state.
pub struct AlgorithmState {
    /// Eq. 3: coefficient matrix c ∈ R^{2×n}
    pub coefficients: CoefficientMatrix,

    /// Collocation points Z = {z_j} (Eq. 4), sampled on a dense grid
    pub collocation_points: Vec<Vector2<f64>>,

    /// Active set Z' ⊂ Z (Algorithm 1), stored as indices into collocation_points
    pub active_set: Vec<usize>,

    /// Stabilization set Z'' (Algorithm 1: "farthest point samples"),
    /// stored as indices into collocation_points
    pub stable_set: Vec<usize>,

    /// Frame vectors d_i (Eq. 27), one per collocation point
    pub frames: Vec<Vector2<f64>>,

    /// K_high, K_low (Section 5 "Activation of constraints")
    /// Default: K_high = 0.1 + 0.9*K, K_low = 0.5 + 0.5*K
    pub k_high: f64,
    pub k_low: f64,

    /// Domain interior mask: `true` if the collocation point falls inside
    /// the domain Ω (contour polygon).  Points outside the domain are kept
    /// in the rectangular grid (so local-maxima detection on the grid still
    /// works), but they are never added to the active/stable sets and are
    /// excluded from ARAP regularisation sampling.
    ///
    /// Paper Section 4: "consider all the points from a surrounding uniform
    /// grid that fall inside the domain"
    pub domain_mask: Vec<bool>,

    /// Precomputed basis function evaluations at collocation points
    pub precomputed: Option<PrecomputedData>,
}

/// Precomputed data for efficiency (computed once in Algorithm 1 "if first step").
pub struct PrecomputedData {
    /// f_i(z) for all z ∈ Z, shape: (num_collocation, num_basis)
    pub phi: DMatrix<f64>,

    /// ∂f_i/∂x at each z, shape: (num_collocation, num_basis)
    pub grad_phi_x: DMatrix<f64>,

    /// ∂f_i/∂y at each z, shape: (num_collocation, num_basis)
    pub grad_phi_y: DMatrix<f64>,

    /// Biharmonic quadratic form matrix (Eq. 31, numerically integrated)
    /// Shape: (2*n, 2*n) where n = num_basis
    pub biharmonic_matrix: Option<DMatrix<f64>>,
}

/// Errors from the solver.
#[derive(Debug)]
pub enum SolverError {
    /// The SOCP solver did not find a feasible solution
    Infeasible(String),
    /// Numerical issues in problem construction
    NumericalError(String),
    /// Solver returned an unexpected status
    SolverFailed(String),
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::Infeasible(msg) => write!(f, "SOCP infeasible: {}", msg),
            SolverError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
            SolverError::SolverFailed(msg) => write!(f, "Solver failed: {}", msg),
        }
    }
}

impl std::error::Error for SolverError {}

/// Errors from the algorithm.
#[derive(Debug)]
pub enum AlgorithmError {
    Solver(SolverError),
    InvalidInput(String),
}

impl std::fmt::Display for AlgorithmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgorithmError::Solver(e) => write!(f, "Algorithm solver error: {}", e),
            AlgorithmError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for AlgorithmError {}

impl From<SolverError> for AlgorithmError {
    fn from(e: SolverError) -> Self {
        AlgorithmError::Solver(e)
    }
}
