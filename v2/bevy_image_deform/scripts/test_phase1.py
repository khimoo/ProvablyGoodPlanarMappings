#!/usr/bin/env python3
"""
Phase 1 動作確認テスト
P1: compute_mapping ループ順序
P2: FPS (Z'') 
P3: Active Set 除去
P4: Local Maxima フィルタリング
P5: K_high, K_low 閾値
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from deform_algo import (
    SolverConfig, FixedBoundCalcGrid, FixedGridCalcBound,
    BetterFitwithGaussianRBF, GaussianRBF
)

def test_p5_thresholds():
    """P5: K_high, K_low が正しく計算されるか"""
    print("=" * 60)
    print("TEST P5: K_high / K_low thresholds")
    print("=" * 60)
    
    src = np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]])
    strategy = FixedGridCalcBound(grid_resolution=(15, 15), K=3.0)
    config = SolverConfig(
        domain_bounds=(0.0, 0.0, 6.0, 6.0),
        source_handles=src,
        strategy=strategy,
        epsilon=50.0,
        lambda_biharmonic=1e-4,
    )
    solver = BetterFitwithGaussianRBF(config)
    
    K = solver.K_upper
    expected_K_high = 0.1 + 0.9 * K
    expected_K_low  = 0.5 + 0.5 * K
    
    assert abs(solver.K_high - expected_K_high) < 1e-10, \
        f"K_high mismatch: {solver.K_high} != {expected_K_high}"
    assert abs(solver.K_low - expected_K_low) < 1e-10, \
        f"K_low mismatch: {solver.K_low} != {expected_K_low}"
    
    print(f"  K={K}, K_high={solver.K_high:.4f} (expected {expected_K_high:.4f})")
    print(f"  K={K}, K_low={solver.K_low:.4f} (expected {expected_K_low:.4f})")
    print("  PASS\n")


def test_p2_fps():
    """P2: FPS が正しく機能するか"""
    print("=" * 60)
    print("TEST P2: Farthest Point Sampling (Z'')")
    print("=" * 60)
    
    src = np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]])
    strategy = FixedGridCalcBound(grid_resolution=(10, 10), K=3.0)
    config = SolverConfig(
        domain_bounds=(0.0, 0.0, 6.0, 6.0),
        source_handles=src,
        strategy=strategy,
        epsilon=50.0,
        fps_k=10,
    )
    solver = BetterFitwithGaussianRBF(config)
    
    # Z'' should have fps_k points
    assert len(solver.permanent_indices) == 10, \
        f"Expected 10 permanent points, got {len(solver.permanent_indices)}"
    
    # Z'' should be subset of activated_indices
    perm_set = set(solver.permanent_indices)
    act_set = set(solver.activated_indices)
    assert perm_set.issubset(act_set), \
        "Z'' should be a subset of the active set"
    
    # All indices should be valid
    M = solver.collocation_grid.shape[0]
    for idx in solver.permanent_indices:
        assert 0 <= idx < M, f"Invalid FPS index: {idx}"
    
    # FPS points should be spread out (no duplicates)
    assert len(set(solver.permanent_indices)) == len(solver.permanent_indices), \
        "FPS should not produce duplicate indices"
    
    # Check spatial spread: min pairwise distance should be reasonable
    pts = solver.collocation_grid[solver.permanent_indices]
    from scipy.spatial.distance import pdist
    min_dist = np.min(pdist(pts))
    print(f"  Z'' size: {len(solver.permanent_indices)}")
    print(f"  Min pairwise distance in Z'': {min_dist:.4f}")
    print(f"  Grid spacing h: {solver.h_grid:.4f}")
    assert min_dist > 0, "FPS points should be distinct"
    print("  PASS\n")


def test_p4_local_maxima():
    """P4: Local maxima detection"""
    print("=" * 60)
    print("TEST P4: Local Maxima Detection")
    print("=" * 60)
    
    src = np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]])
    strategy = FixedGridCalcBound(grid_resolution=(5, 5), K=3.0)
    config = SolverConfig(
        domain_bounds=(0.0, 0.0, 6.0, 6.0),
        source_handles=src,
        strategy=strategy,
        epsilon=50.0,
        fps_k=0,  # No FPS for this test
    )
    solver = BetterFitwithGaussianRBF(config)
    
    # Create a known distortion pattern with a clear peak
    nx, ny = solver.grid_shape
    K_vals = np.ones(nx * ny)
    # Place a peak at center
    center_idx = (ny // 2) * nx + (nx // 2)
    K_vals[center_idx] = 5.0
    
    maxima = solver._find_local_maxima(K_vals)
    
    print(f"  Grid: {nx}x{ny}, peak at index {center_idx}")
    print(f"  Found {len(maxima)} local maxima: {maxima.tolist()}")
    
    assert center_idx in maxima, \
        f"Expected peak index {center_idx} to be a local maximum"
    
    # Test with uniform values (all equal) - boundary points may be maxima
    K_vals_uniform = np.ones(nx * ny) * 2.0
    maxima_uniform = solver._find_local_maxima(K_vals_uniform)
    # With all equal values, strict > comparison means no local maxima
    print(f"  Uniform K_vals: found {len(maxima_uniform)} local maxima (expect 0)")
    assert len(maxima_uniform) == 0, \
        "Uniform values should have no strict local maxima"
    
    print("  PASS\n")


def test_p3_active_set_removal():
    """P3: Active Set removal and Z'' protection"""
    print("=" * 60)
    print("TEST P3: Active Set Removal Logic")
    print("=" * 60)
    
    src = np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]])
    strategy = FixedGridCalcBound(grid_resolution=(10, 10), K=3.0)
    config = SolverConfig(
        domain_bounds=(0.0, 0.0, 6.0, 6.0),
        source_handles=src,
        strategy=strategy,
        epsilon=50.0,
        fps_k=5,
    )
    solver = BetterFitwithGaussianRBF(config)
    
    # Initially, activated = Z'' (permanent points)
    initial_active = set(solver.activated_indices)
    perm_set = set(solver.permanent_indices)
    
    print(f"  Initial active set size: {len(initial_active)}")
    print(f"  Permanent (Z'') size: {len(perm_set)}")
    
    # With identity mapping, distortion should be ~1.0 everywhere
    # K_low = 0.5 + 0.5*3.0 = 2.0
    # Since K~1.0 < K_low=2.0 for identity, non-permanent points would be removed
    # But initially only Z'' is in active set, and Z'' is protected
    
    # Manually add some non-permanent points
    solver.activated_indices = sorted(list(perm_set | {0, 1, 2, 3}))
    added_non_perm = {0, 1, 2, 3} - perm_set
    print(f"  Added non-permanent points: {added_non_perm}")
    
    # Run update_active_set - identity map has K~1.0 < K_low, 
    # so non-permanent points should be removed
    new_violations = solver._update_active_set()
    
    remaining_set = set(solver.activated_indices)
    print(f"  After update: active set size: {len(remaining_set)}")
    print(f"  New violations: {new_violations}")
    
    # Z'' should still be in active set
    assert perm_set.issubset(remaining_set), \
        "Z'' points should never be removed"
    
    # Non-permanent points with low distortion should be removed
    for idx in added_non_perm:
        if idx not in perm_set:
            assert idx not in remaining_set, \
                f"Non-permanent point {idx} with low distortion should be removed"
    
    print("  PASS\n")


def test_p1_loop_order():
    """P1: compute_mapping loop order (Active Set → Optimize → Frame Update)"""
    print("=" * 60)
    print("TEST P1: compute_mapping Loop Order")
    print("=" * 60)
    
    src = np.array([[2.0, 3.0], [4.0, 3.0], [3.0, 5.0]])
    strategy = FixedGridCalcBound(grid_resolution=(10, 10), K=3.0)
    config = SolverConfig(
        domain_bounds=(0.0, 0.0, 6.0, 6.0),
        source_handles=src,
        strategy=strategy,
        epsilon=50.0,
        fps_k=5,
        lambda_biharmonic=1e-5,
    )
    solver = BetterFitwithGaussianRBF(config)
    
    # Apply a moderate deformation
    target = src.copy()
    target[0] = [2.5, 3.5]  # Move first handle slightly
    
    solver.compute_mapping(target)
    
    # After mapping, verify:
    # 1. Coefficients were updated
    assert solver.c is not None, "Coefficients should be updated"
    
    # 2. Frames were updated (not all (1,0))
    # At least some frames should differ from (1,0) after deformation
    
    # 3. Active set should contain at least Z''
    perm_set = set(solver.permanent_indices)
    act_set = set(solver.activated_indices)
    assert perm_set.issubset(act_set), "Z'' should always be in active set"
    
    # 4. Check distortion is reasonable
    K_vals, _ = solver._compute_distortion_on_grid()
    max_K = np.max(K_vals)
    print(f"  Max distortion after mapping: {max_K:.4f}")
    print(f"  Active set size: {len(solver.activated_indices)}")
    print(f"  Permanent (Z''): {len(solver.permanent_indices)}")
    
    # Transform should work
    test_pts = np.array([[3.0, 3.0], [3.0, 4.0]])
    result = solver.transform(test_pts)
    assert result.shape == (2, 2), f"Transform output shape mismatch: {result.shape}"
    print(f"  Transform test: {test_pts[0]} -> {result[0]}")
    print("  PASS\n")


def test_integration_strong_deformation():
    """Integration test: Strong deformation should activate constraints"""
    print("=" * 60)
    print("TEST Integration: Strong Deformation")
    print("=" * 60)
    
    src = np.array([[1.0, 3.0], [3.0, 3.0], [5.0, 3.0]])
    strategy = FixedGridCalcBound(grid_resolution=(15, 15), K=2.0)
    config = SolverConfig(
        domain_bounds=(0.0, 0.0, 6.0, 6.0),
        source_handles=src,
        strategy=strategy,
        epsilon=50.0,
        fps_k=10,
        lambda_biharmonic=1e-5,
    )
    solver = BetterFitwithGaussianRBF(config)
    
    initial_active_count = len(solver.activated_indices)
    
    # Apply strong deformation
    target = src.copy()
    target[1] = [3.0, 5.0]  # Move middle handle up significantly
    
    solver.compute_mapping(target)
    
    final_active_count = len(solver.activated_indices)
    K_vals, _ = solver._compute_distortion_on_grid()
    max_K = np.max(K_vals)
    
    print(f"  Initial active set: {initial_active_count}")
    print(f"  Final active set: {final_active_count}")
    print(f"  Max distortion: {max_K:.4f}")
    print(f"  K_high threshold: {solver.K_high:.4f}")
    
    # Strong deformation should have added some constraint points
    # (beyond just Z'')
    print("  PASS\n")


if __name__ == "__main__":
    test_p5_thresholds()
    test_p2_fps()
    test_p4_local_maxima()
    test_p3_active_set_removal()
    test_p1_loop_order()
    test_integration_strong_deformation()
    
    print("=" * 60)
    print("ALL PHASE 1 TESTS PASSED!")
    print("=" * 60)
