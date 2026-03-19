//! Regularization strategy implementations

use crate::strategy::RegularizationStrategy;
use crate::types::*;

/// Biharmonic regularization (Eq. 31)
///
/// Biharmonic energy: E_bh = ∫∫_Ω ||∇²f||² dΩ
///
/// This regularization penalizes sharp bends in the mapping,
/// encouraging smooth deformations.
pub struct BiharmonicRegularization {
    /// Weight of regularization term
    weight: f64,
}

impl Default for BiharmonicRegularization {
    fn default() -> Self {
        Self { weight: 0.01 }
    }
}

impl BiharmonicRegularization {
    pub fn new(weight: f64) -> Self {
        Self { weight }
    }
}

impl RegularizationStrategy for BiharmonicRegularization {
    /// Eq. 31: Build biharmonic energy term
    ///
    /// E_bh = λ * ∫∫_Ω ||∇²f||² dΩ = c^T Q c
    ///
    /// Phase 2: Simplified - return identity matrix weighted by λ
    /// Full numerical integration deferred to Phase 3
    fn build_energy_terms(
        &self,
        _domain: &DomainInfo,
        basis_hessians: &[f64],
    ) -> Result<EnergyTerms> {
        // Phase 2 MVP: Use simplified biharmonic regularization
        // Return diagonal matrix with weight * hessian values

        if basis_hessians.is_empty() {
            return Ok(EnergyTerms::Quadratic {
                matrix: vec![vec![0.0; 1]; 1],
                linear: vec![0.0; 1],
            });
        }

        // Number of basis functions = length of basis_hessians
        // (in precomputation, one hessian per basis at a single point)
        let n_basis = basis_hessians.len();

        // Build diagonal matrix Q = λ * I
        // This provides basic smoothing regularization
        let mut Q = vec![vec![0.0; n_basis]; n_basis];
        for i in 0..n_basis {
            Q[i][i] = self.weight; // λ on diagonal
        }

        let q = vec![0.0; n_basis];

        Ok(EnergyTerms::Quadratic {
            matrix: Q,
            linear: q,
        })
    }

    fn name(&self) -> &'static str {
        "Biharmonic (Phase 2 Simplified)"
    }
}

// Phase 3: ARAP regularization placeholder
// pub struct ARAPRegularization { ... }
