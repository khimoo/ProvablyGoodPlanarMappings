//! Distortion strategy implementations

use crate::strategy::DistortionStrategy;
use crate::types::*;

/// Isometric distortion strategy (Eq. 10-11, 14-17)
///
/// Isometric distortion: σ_iso(f) = max{σ_max(∇f), 1/σ_min(∇f)}
/// Where σ_max, σ_min are singular values of the Jacobian ∇f.
///
/// K_high and K_low are thresholds for active set management (Eq. 14-17).
pub struct IsometricStrategy {
    /// K_high: Threshold for activating constraints
    /// When σ > K_high, add this point to active set
    pub k_high: f64,

    /// K_low: Threshold for deactivating constraints
    /// When σ < K_low, remove from active set
    pub k_low: f64,
}

impl Default for IsometricStrategy {
    fn default() -> Self {
        // Section 5 "Activation of constraints": Default values
        Self {
            k_high: 1.1,
            k_low: 1.05,
        }
    }
}

impl IsometricStrategy {
    pub fn new(k_high: f64, k_low: f64) -> Self {
        Self { k_high, k_low }
    }
}

impl DistortionStrategy for IsometricStrategy {
    /// Eq. 10-11: Compute σ_iso from Jacobian singular values
    ///
    /// σ_iso = max{σ_max, 1/σ_min}
    /// This ensures both:
    /// - Stretching is bounded (σ_max not too large)
    /// - Compression is bounded (1/σ_min not too large)
    fn compute_distortion(&self, jacobian: Mat2) -> f64 {
        // Singular value decomposition
        let svd = jacobian.svd(true, true);
        let singular_values = svd.singular_values;

        let sigma_max = singular_values[0]; // Largest singular value
        let sigma_min = singular_values[1]; // Smallest singular value

        // Eq. 10: D_iso = max{σ_max, 1/σ_min}
        if sigma_min < 1e-12 {
            // Near-singular case: very large distortion
            f64::INFINITY
        } else {
            sigma_max.max(1.0 / sigma_min)
        }
    }

    /// Eq. 14-17: K_high and K_low for active set management
    fn get_activation_threshold(&self) -> (f64, f64) {
        (self.k_high, self.k_low)
    }

    /// Build SOCP constraints (Eq. 21-23, 26)
    ///
    /// For each active point x_i in the active set:
    ///   ||J_S(x_i)|| ≤ t_i                 (23a)
    ///   ||J_A(x_i)|| ≤ s_i                 (23b)
    ///   t_i + s_i ≤ K                      (23c)
    ///   J_S(x_i) · d_i - s_i ≥ 1/K        (26)
    fn build_constraints(
        &self,
        active_set: &ActiveSetInfo,
        _basis_evals: &[f64],
        basis_grads: &[Vec2],
    ) -> Result<Vec<ConeConstraint>> {
        let mut constraints = Vec::new();

        // Iterate over active collocation points
        for (idx, &center) in active_set.centers.iter().enumerate() {
            if !active_set.is_active[idx] {
                continue; // Skip inactive points
            }

            // Extract basis gradients for this point
            // basis_grads[idx * n_coeff : (idx+1) * n_coeff] are gradients at center
            let n_coeff = basis_grads.len() / active_set.centers.len();
            let start = idx * n_coeff;
            let end = start + n_coeff;

            if end > basis_grads.len() {
                continue; // Skip if insufficient gradient data
            }

            let grads_at_point = basis_grads[start..end].to_vec();

            // Create cone constraint for this active point
            // Phase 2: Simplified - just use K_high as bound
            // Full Eq. 21-26 formulation deferred to Phase 3
            constraints.push(ConeConstraint {
                center,
                basis_grads: grads_at_point,
                bound: self.k_high, // K from Eq. 23c
            });
        }

        Ok(constraints)
    }

    fn name(&self) -> &'static str {
        "Isometric"
    }
}

// Phase 3: Conformal strategy placeholder
// pub struct ConformalStrategy { ... }
