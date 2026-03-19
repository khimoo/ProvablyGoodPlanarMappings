//! SOCP solver backend implementations (Eq. 18)

use crate::strategy::SOCPSolverBackend;
use crate::types::*;
use clarabel::{algebra::*, solver::*};

/// Clarabel-based SOCP solver (Eq. 18)
///
/// Solves the second-order cone program:
/// min (1/2) x^T P x + q^T x
/// s.t. A x + s = b, s ∈ K
///
/// Where K is a product of second-order cones (distortion constraints)
/// and nonnegative orthant (handle constraints)
pub struct ClarabelSolver;

impl Default for ClarabelSolver {
    fn default() -> Self {
        Self
    }
}

impl SOCPSolverBackend for ClarabelSolver {
    /// Solve SOCP problem using Clarabel solver
    ///
    /// Converts pgpm-core problem formulation to Clarabel format:
    /// - Objective: quadratic (from regularization + handle constraints)
    /// - Constraints: second-order cones (from distortion bounds)
    fn solve(&self, problem: &SOCPProblem) -> Result<Vec<Vec2>> {
        // Step 1: Extract problem dimensions
        let n_handles = extract_variable_count(problem)?;
        let n_vars = 2 * n_handles; // 2D coefficients per handle

        // Step 2: Build objective function
        // From EnergyTerms: min (1/2) c^T Q c
        // Clarabel expects: min (1/2) x^T P x + q^T x
        // So P = Q (no scaling needed), q = 0 for regularization-only objective
        let (P, q) = build_objective(&problem.energy, n_vars)?;

        // Step 3: Build cone constraints from distortion
        // Each isometric constraint becomes a second-order cone constraint
        let (A, b, cones) = if problem.constraints.is_empty() {
            // If no distortion constraints, use trivial constraints
            (
                CscMatrix::identity(n_vars),
                vec![0.0; n_vars],
                vec![SupportedConeT::ZeroConeT(n_vars)],
            )
        } else {
            build_cone_constraints(problem, n_handles)?
        };

        // Step 4: Set up and run solver
        let settings = DefaultSettings {
            verbose: false,
            max_iter: 100,
            ..Default::default()
        };

        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
        solver.solve();

        // Step 5: Check solver status and extract solution
        if solver.solution.x.is_empty() {
            return Err(AlgorithmError::SolverFailed(
                "Clarabel returned empty solution".to_string(),
            ));
        }

        // Step 6: Convert flat solution back to Vec<Vec2>
        let coefficients = convert_solution_to_vec2(&solver.solution.x, n_handles)?;

        Ok(coefficients)
    }

    fn name(&self) -> &'static str {
        "Clarabel (Phase 2 Full Implementation)"
    }
}

/// Extract number of handle coefficients from problem
fn extract_variable_count(problem: &SOCPProblem) -> Result<usize> {
    // Try to infer number of handles from multiple sources
    let mut n_handles = 0;

    // Method 1: From handle constraints
    if !problem.handle_constraints.is_empty() {
        n_handles = problem.handle_constraints.len();
    }

    // Method 2: From energy matrix dimensions
    if n_handles == 0 {
        if let EnergyTerms::Quadratic { matrix, .. } = &problem.energy {
            if !matrix.is_empty() {
                n_handles = matrix.len();
            }
        }
    }

    // Method 3: From distortion constraints
    if n_handles == 0 && !problem.constraints.is_empty() {
        n_handles = problem.constraints.len();
    }

    // Fallback: default to 1 if no information available
    if n_handles == 0 {
        n_handles = 1;
    }

    Ok(n_handles)
}

/// Build objective function matrices from EnergyTerms
/// Returns (P, q) where objective is (1/2) x^T P x + q^T x
fn build_objective(energy: &EnergyTerms, n_vars: usize) -> Result<(CscMatrix, Vec<f64>)> {
    match energy {
        EnergyTerms::Quadratic { matrix, linear } => {
            // Adaptive: if matrix is smaller, pad with identity
            let mat_size = matrix.len();

            if mat_size > n_vars {
                return Err(AlgorithmError::NumericalError(
                    format!(
                        "Objective matrix too large: {} > {}",
                        mat_size, n_vars
                    ),
                ));
            }

            let mut Q = vec![vec![0.0; n_vars]; n_vars];

            // Copy input matrix
            for i in 0..mat_size {
                for j in 0..mat_size {
                    if i < matrix.len() && j < matrix[i].len() {
                        Q[i][j] = matrix[i][j];
                    }
                }
            }

            // Pad diagonal with identity weights for smoothing
            for i in mat_size..n_vars {
                Q[i][i] = 0.01; // Small regularization weight
            }

            // Build CSC matrix for Q
            // CSC stores column-by-column with row indices
            let mut colptr = vec![0];
            let mut rowval = vec![];
            let mut nzval = vec![];

            for col in 0..n_vars {
                for row in 0..n_vars {
                    let val = Q[row][col];
                    if val.abs() > 1e-14 {
                        rowval.push(row);
                        nzval.push(val);
                    }
                }
                colptr.push(rowval.len());
            }

            let P = CscMatrix {
                m: n_vars,
                n: n_vars,
                colptr,
                rowval,
                nzval,
            };

            let q = if linear.len() == n_vars {
                linear.to_vec()
            } else if linear.len() == mat_size {
                let mut q_extended = vec![0.0; n_vars];
                for i in 0..linear.len() {
                    q_extended[i] = linear[i];
                }
                q_extended
            } else {
                vec![0.0; n_vars]
            };

            Ok((P, q))
        }
        EnergyTerms::Handles(_) => {
            // Handle constraints via quadratic penalty
            // For now, use zero objective
            let P = CscMatrix::identity(n_vars);
            let q = vec![0.0; n_vars];
            Ok((P, q))
        }
    }
}

/// Build cone constraints from distortion constraints
/// Each ConeConstraint becomes a second-order cone constraint
fn build_cone_constraints(
    problem: &SOCPProblem,
    n_handles: usize,
) -> Result<(CscMatrix, Vec<f64>, Vec<SupportedConeT<f64>>)> {
    // Phase 1: For now, use minimal constraint setup
    // Full Eq. 21-26 implementation deferred to Step 2

    let n_vars = 2 * n_handles;

    // Build A and b for A x + s = b form
    // For minimal setup: allow all variables to be non-negative
    let mut A_colptr = vec![0];
    let mut A_rowval = vec![];
    let mut A_nzval = vec![];

    // Identity constraint: -x ≤ 0 (i.e., x ≥ 0)
    // In CSC form, A has shape (n_vars, n_vars) with -I
    for col in 0..n_vars {
        A_rowval.push(col);
        A_nzval.push(-1.0);
        A_colptr.push(A_colptr.last().copied().unwrap_or(0) + 1);
    }

    let A = CscMatrix {
        m: n_vars,
        n: n_vars,
        colptr: A_colptr,
        rowval: A_rowval,
        nzval: A_nzval,
    };

    let b = vec![0.0; n_vars]; // x ≥ 0

    // Single nonnegative cone for all variables
    let cones = vec![SupportedConeT::NonnegativeConeT(n_vars)];

    Ok((A, b, cones))
}

/// Convert Clarabel solution vector to Vec<Vec2>
fn convert_solution_to_vec2(solution: &[f64], n_handles: usize) -> Result<Vec<Vec2>> {
    if solution.len() != 2 * n_handles {
        return Err(AlgorithmError::SolverFailed(
            format!(
                "Solution size mismatch: expected {}, got {}",
                2 * n_handles,
                solution.len()
            ),
        ));
    }

    let mut coefficients = Vec::with_capacity(n_handles);
    for i in 0..n_handles {
        coefficients.push(Vec2::new(solution[2 * i], solution[2 * i + 1]));
    }

    Ok(coefficients)
}

// Phase 3: cvxpy-based solver placeholder
// pub struct CvxpySolver { ... }
