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

/// Eq. 5: Bounding box of domain Ω.
#[derive(Debug, Clone)]
pub struct DomainBounds {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
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

/// SOCP solver numerical tuning parameters.
///
/// These parameters control the solver's numerical behaviour and
/// resource limits.  They are **implementation-specific** and do not
/// appear in the paper's formulation.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Diagonal regularization added to the P matrix for coefficient
    /// variables (first 2n entries).  Prevents near-singular KKT systems
    /// when the regularization weight λ is very small.
    pub p_reg_coefficient: f64,
    /// Diagonal regularization for auxiliary variables (r, t, s).
    pub p_reg_auxiliary: f64,
    /// Maximum grid resolution (points per side) for Strategy 2 refinement.
    /// Paper Section 6 uses up to 6000².  Set according to available memory
    /// and time budget.
    pub max_refinement_resolution: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            p_reg_coefficient: 1e-6,
            p_reg_auxiliary: 1e-8,
            max_refinement_resolution: 1000,
        }
    }
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

    /// Grid dimensions for local maxima search (Section 5)
    pub grid_width: usize,
    pub grid_height: usize,
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
    /// Strategy 2 requires a grid resolution exceeding the configured
    /// maximum (`SolverConfig::max_refinement_resolution`).
    /// The caller (e.g. UI) should ask the user whether to proceed
    /// with the required resolution.
    ResolutionExceeded { required: usize, max: usize },
}

impl std::fmt::Display for AlgorithmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgorithmError::Solver(e) => write!(f, "Algorithm solver error: {}", e),
            AlgorithmError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            AlgorithmError::ResolutionExceeded { required, max } => write!(
                f,
                "Strategy 2 requires resolution {} which exceeds max {}",
                required, max
            ),
        }
    }
}

impl std::error::Error for AlgorithmError {}

impl From<SolverError> for AlgorithmError {
    fn from(e: SolverError) -> Self {
        AlgorithmError::Solver(e)
    }
}

/// Algorithm 1 input parameters (distortion-type-agnostic).
///
/// This struct contains the parameters that are independent of the
/// distortion measure (isometric vs conformal). The distortion-specific
/// behaviour is handled by the `DistortionPolicy` trait.
#[derive(Debug, Clone)]
pub struct MappingParams {
    /// Distortion upper bound K (Eq. 4: D(z_j) ≤ K)
    pub k_bound: f64,

    /// Regularization weight λ (Eq. 1, 18)
    pub lambda_reg: f64,

    /// Regularization type and mixing weights
    pub regularization: RegularizationType,
}

/// Information returned from each algorithm step.
#[derive(Debug)]
pub struct StepInfo {
    /// Maximum distortion over all collocation points (evaluated before SOCP solve)
    pub max_distortion: f64,
    /// Number of points in the active set Z'
    pub active_set_size: usize,
    /// Number of points in the stable set Z''
    pub stable_set_size: usize,
    /// Algorithm 1 convergence: max_distortion ≤ K and active set unchanged.
    ///
    /// When `true`, the pre-solve distortion was already within the bound
    /// and no new constraint points were needed, meaning the previous
    /// SOCP solution satisfies the distortion bound at all collocation points.
    pub converged: bool,
}
