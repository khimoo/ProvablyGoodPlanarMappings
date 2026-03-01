//! Active set management.
//!
//! Paper references:
//! - Algorithm 1: Initialization and active set update
//! - Section 5 "Activation of constraints":
//!   "the local maxima of the distortion are found"
//!   "only points where D(z) > K_high are added"
//!   "points where D(z) < K_low are removed"

use crate::types::AlgorithmState;
use nalgebra::Vector2;

/// Algorithm 1: Update the active set Z'.
///
/// Follows the paper exactly:
/// 1. Find Z_max = local maxima of distortion on the grid
/// 2. For z ∈ Z_max with D(z) > K_high: insert z into Z'
/// 3. For z ∈ Z' with D(z) < K_low: remove z from Z'
///
/// **IMPORTANT**: No fold-over prevention, no σ checks, no size limits.
/// The paper states local maxima filtering is sufficient.
pub fn update_active_set(
    state: &mut AlgorithmState,
    distortions: &[f64],
    grid_width: usize,
    grid_height: usize,
) {
    // Step 1: Find local maxima on the grid
    let local_maxima = find_local_maxima(distortions, grid_width, grid_height);

    // Step 2: Add local maxima exceeding K_high
    // "foreach z ∈ Z_max such that D(z) > K_high do insert z to Z'"
    // Paper Section 4: only points that fall inside the domain are constrained.
    for &idx in &local_maxima {
        if distortions[idx] > state.k_high && state.domain_mask[idx] {
            if !state.active_set.contains(&idx) {
                state.active_set.push(idx);
            }
        }
    }

    // Step 3: Remove points below K_low
    // "foreach z ∈ Z' such that D(z) < K_low do remove z from Z'"
    state.active_set.retain(|&idx| distortions[idx] >= state.k_low);
}

/// Detect local maxima on a rectangular grid (8-neighbor comparison).
///
/// Section 5: "the local maxima of the distortion are found"
/// "the collocation points are sampled on a dense rectangular grid"
///
/// A point is a local maximum if its distortion is strictly greater than
/// all of its neighbors (up to 8 on a grid).
fn find_local_maxima(
    distortions: &[f64],
    grid_width: usize,
    grid_height: usize,
) -> Vec<usize> {
    let mut maxima = Vec::new();

    for row in 0..grid_height {
        for col in 0..grid_width {
            let idx = row * grid_width + col;
            let val = distortions[idx];

            let mut is_max = true;

            // Check all 8 neighbors
            for dr in [-1i32, 0, 1] {
                for dc in [-1i32, 0, 1] {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    if nr >= 0
                        && nr < grid_height as i32
                        && nc >= 0
                        && nc < grid_width as i32
                    {
                        let nidx = nr as usize * grid_width + nc as usize;
                        if distortions[nidx] >= val {
                            is_max = false;
                            break;
                        }
                    }
                }
                if !is_max {
                    break;
                }
            }

            if is_max {
                maxima.push(idx);
            }
        }
    }

    maxima
}

/// Algorithm 1: "Initialize set Z'' with farthest point samples"
///
/// Farthest Point Sampling (FPS) to select k evenly-spaced points
/// from the collocation points **that lie inside the domain**.
///
/// Section 5: "we may keep a small subset of equally spread
/// collocation points always active"
pub fn initialize_stable_set(
    collocation_points: &[Vector2<f64>],
    k: usize,
    domain_mask: &[bool],
) -> Vec<usize> {
    let n = collocation_points.len();
    // Candidate indices: only domain-interior points
    let candidates: Vec<usize> = (0..n).filter(|&i| domain_mask[i]).collect();
    if k == 0 || candidates.is_empty() {
        return Vec::new();
    }
    if k >= candidates.len() {
        return candidates;
    }

    let mut selected = Vec::with_capacity(k);
    let mut min_dists = vec![f64::INFINITY; n];

    // Start from the first candidate
    selected.push(candidates[0]);

    for _ in 1..k {
        // Update minimum distances based on the last selected point
        let last = *selected.last().unwrap();
        let last_pt = collocation_points[last];
        for &i in &candidates {
            let d = (collocation_points[i] - last_pt).norm_squared();
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
        // Set distance of already-selected points to 0 to avoid re-selection
        for &s in &selected {
            min_dists[s] = 0.0;
        }

        // Select the candidate with maximum minimum distance
        let best = candidates.iter()
            .copied()
            .filter(|i| !selected.contains(i))
            .max_by(|&a, &b| min_dists[a].partial_cmp(&min_dists[b]).unwrap())
            .unwrap();

        selected.push(best);
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;

    #[test]
    fn test_local_maxima_simple() {
        // 3x3 grid with a single peak in the center
        #[rustfmt::skip]
        let distortions = vec![
            1.0, 1.0, 1.0,
            1.0, 5.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let maxima = find_local_maxima(&distortions, 3, 3);
        assert_eq!(maxima, vec![4]); // center
    }

    #[test]
    fn test_local_maxima_corner() {
        // Corner peak
        #[rustfmt::skip]
        let distortions = vec![
            5.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let maxima = find_local_maxima(&distortions, 3, 3);
        assert_eq!(maxima, vec![0]);
    }

    #[test]
    fn test_local_maxima_plateau_excluded() {
        // Plateau: no strict local maximum
        #[rustfmt::skip]
        let distortions = vec![
            1.0, 1.0, 1.0,
            1.0, 5.0, 5.0,
            1.0, 1.0, 1.0,
        ];
        let maxima = find_local_maxima(&distortions, 3, 3);
        // Neither (1,1) nor (1,2) is a strict local max because they're equal
        assert!(maxima.is_empty());
    }

    #[test]
    fn test_active_set_update() {
        let mut state = crate::types::AlgorithmState {
            coefficients: nalgebra::DMatrix::zeros(2, 1),
            collocation_points: Vec::new(),
            active_set: Vec::new(),
            stable_set: Vec::new(),
            frames: Vec::new(),
            k_high: 2.8, // 0.1 + 0.9*3
            k_low: 2.0,   // 0.5 + 0.5*3
            domain_mask: vec![true; 9],
            precomputed: None,
        };

        // 3x3 grid, center has high distortion
        #[rustfmt::skip]
        let distortions = vec![
            1.0, 1.0, 1.0,
            1.0, 3.5, 1.0,
            1.0, 1.0, 1.0,
        ];

        update_active_set(&mut state, &distortions, 3, 3);
        // Center (idx=4) is a local max and D=3.5 > K_high=2.8
        assert!(state.active_set.contains(&4));
    }

    #[test]
    fn test_active_set_removal() {
        let mut state = crate::types::AlgorithmState {
            coefficients: nalgebra::DMatrix::zeros(2, 1),
            collocation_points: Vec::new(),
            active_set: vec![4],
            stable_set: Vec::new(),
            frames: Vec::new(),
            k_high: 2.8,
            k_low: 2.0,
            domain_mask: vec![true; 9],
            precomputed: None,
        };

        // Now distortion at 4 drops below K_low
        #[rustfmt::skip]
        let distortions = vec![
            1.0, 1.0, 1.0,
            1.0, 1.5, 1.0,
            1.0, 1.0, 1.0,
        ];

        update_active_set(&mut state, &distortions, 3, 3);
        // Point 4 has D=1.5 < K_low=2.0, should be removed
        assert!(!state.active_set.contains(&4));
    }

    #[test]
    fn test_fps_basic() {
        // 4 corners of a unit square
        let points = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(0.0, 1.0),
            Vector2::new(1.0, 1.0),
        ];
        let mask = vec![true; 4];

        let selected = initialize_stable_set(&points, 4, &mask);
        assert_eq!(selected.len(), 4);
        // All 4 should be selected
        for i in 0..4 {
            assert!(selected.contains(&i));
        }
    }

    #[test]
    fn test_fps_selects_spread_points() {
        // Line of points, FPS should pick endpoints and middle
        let points: Vec<Vector2<f64>> = (0..11)
            .map(|i| Vector2::new(i as f64, 0.0))
            .collect();
        let mask = vec![true; 11];

        let selected = initialize_stable_set(&points, 3, &mask);
        assert_eq!(selected.len(), 3);
        // Should include first point (0) and farthest (10), then middle (5)
        assert!(selected.contains(&0));
        assert!(selected.contains(&10));
    }

    #[test]
    fn test_active_set_skips_exterior_points() {
        // 3x3 grid: center is outside domain (mask = false)
        let mut state = crate::types::AlgorithmState {
            coefficients: nalgebra::DMatrix::zeros(2, 1),
            collocation_points: Vec::new(),
            active_set: Vec::new(),
            stable_set: Vec::new(),
            frames: Vec::new(),
            k_high: 2.8,
            k_low: 2.0,
            domain_mask: vec![true, true, true, true, false, true, true, true, true],
            precomputed: None,
        };

        // Center has high distortion but is outside domain
        #[rustfmt::skip]
        let distortions = vec![
            1.0, 1.0, 1.0,
            1.0, 5.0, 1.0,
            1.0, 1.0, 1.0,
        ];

        update_active_set(&mut state, &distortions, 3, 3);
        // Center (idx=4) is a local max and D=5.0 > K_high, but domain_mask[4]=false
        assert!(!state.active_set.contains(&4),
            "Exterior point should not be added to active set");
    }

    #[test]
    fn test_fps_skips_exterior_points() {
        // 5 points, but point 2 is outside domain
        let points = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(2.0, 0.0), // outside
            Vector2::new(3.0, 0.0),
            Vector2::new(4.0, 0.0),
        ];
        let mask = vec![true, true, false, true, true];

        let selected = initialize_stable_set(&points, 3, &mask);
        assert_eq!(selected.len(), 3);
        // Point 2 should never be selected
        assert!(!selected.contains(&2),
            "Exterior point should not be in stable set");
    }
}
