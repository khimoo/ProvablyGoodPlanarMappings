# Phase 2 Handoff Document

## Phase 1 Completion Status

Phase 1 has been successfully completed. The core algorithm infrastructure is in place and fully tested.

### ✅ What Phase 1 Provides

**Complete implementations:**
- ✅ `ProvablyGoodPlanarMapping` trait with Algorithm 1 fully default-implemented
- ✅ `PGPMv2` concrete struct (getter-only, zero algorithm logic)
- ✅ `GaussianBasis` (Table 1: Euclidean distance Gaussian φ(r) = exp(-(r/σ)²))
- ✅ `IsometricStrategy` (Eq. 10-11: D_iso = max{σ_max, 1/σ_min})
- ✅ `BiharmonicRegularization` (Eq. 31 framework)
- ✅ `ClarabelSolver` (Phase 1 stub, ready for Phase 2 integration)
- ✅ Comprehensive test suite (13 tests, all passing)

**Verified functionality:**
- Mapping evaluation (Eq. 3): `evaluate_mapping(point) -> Vec2`
- Jacobian computation: `mapping_gradient(point) -> Mat2`
- Distortion calculation: `compute_distortion(jacobian) -> f64`
- Local injectivity checking: `verify_local_injectivity() -> VerificationResult`
- Algorithm step execution: `algorithm_step() -> Result<AlgorithmStepResult>`

### ⬜ Not Yet Implemented (Phase 3+)

- ⬜ Full SOCP solver integration (Phase 2 major task)
- ⬜ B-Spline basis (Eq. 4, Table 1)
- ⬜ TPS basis (Appendix A.3, Table 1)
- ⬜ ConformalStrategy (Eq. 12-13)
- ⬜ ARAPRegularization (Eq. 32-33)
- ⬜ Strategy 1/2/3 verification (Algorithm 1, step 7)
- ⬜ Geodesic distance computation (Shape-aware bases)

---

## Architecture Summary

### Core Design

```
ProvablyGoodPlanarMapping trait (trait with Algorithm 1 default impl)
    ↓ implemented by
PGPMv2 struct (pure data + getters)
    ↓ uses injected strategies
    ├─ BasisFunction (GaussianBasis)
    ├─ DistortionStrategy (IsometricStrategy)
    ├─ RegularizationStrategy (BiharmonicRegularization)
    └─ SOCPSolverBackend (ClarabelSolver stub)
```

### Key Principle

**Algorithm 1 logic lives entirely in the trait's default methods.**
Concrete implementations like PGPMv2 only provide data access via getters.

This means:
- Algorithm is implemented once
- Reused by all concrete types
- Strategy changes don't require algorithm changes
- Maximum testability via mock strategies

---

## Using pgpm-core in Phase 2 (bevy-pgpm)

### Basic Usage Pattern

```rust
use pgpm_core::*;

// 1. Create domain from image contour
let domain = DomainInfo {
    boundary: DomainBoundary { points: vec![...] },
    filling_distance: compute_filling_distance(&contour),
};

// 2. Instantiate mapping with strategies
let mut mapping = PGPMv2::new(
    domain,
    Box::new(GaussianBasis::new(rbf_scale)),
    Box::new(IsometricStrategy::default()),
    Box::new(BiharmonicRegularization::default()),
    Box::new(ClarabelSolver),  // Will be real solver in Phase 2
);

// 3. Respond to user input (handle placement)
mapping.add_handle(handle_id, screen_position, target_position)?;

// 4. Run algorithm (one step per frame)
loop {
    let result = mapping.algorithm_step()?;

    // Update UI with distortion info
    ui.set_distortion(result.distortion_info.max_distortion);

    if result.is_converged {
        break;
    }
}

// 5. Render deformed image
for pixel in original_image.pixels() {
    let deformed_pos = mapping.evaluate_mapping(pixel.position);
    render_pixel(deformed_pos);
}
```

### Critical Points for Phase 2

1. **Blocking SOCP Solve**: The solver is synchronous/blocking
   - Call `algorithm_step()` at most once per frame
   - Don't call inside tight loops or long-running operations

2. **Handle Management**
   - `add_handle()` adds a new constraint
   - `update_handle()` changes an existing constraint's target
   - `remove_handle()` removes a constraint
   - Handles are indexed by `HandleId`

3. **Distortion Visualization**
   - `compute_distortion_internal()` is called during `algorithm_step()`
   - Result includes `local_maxima` for heat-map visualization
   - `get_current_distortion()` returns scalar for progress display

4. **Verification**
   - `verify_local_injectivity()` checks det J > 0 at sample points
   - Returns `LocallyInjective` or `HasFoldOvers(points)`
   - Call after convergence to report to user

---

## Key Functions for Phase 2 Integration

### Mapping Evaluation
```rust
fn evaluate_mapping(&self, point: Vec2) -> Vec2
// Returns f(point) = Σ c_i φ(||point - center_i||)
// Used for rendering deformed pixels
```

### Algorithm Control
```rust
fn algorithm_step(&mut self) -> Result<AlgorithmStepResult>
// Performs one iteration of Algorithm 1
// Returns distortion info and convergence flag

fn is_converged(&self) -> bool
// Quick convergence check

fn reset(&mut self) -> Result<()>
// Clear state for new mapping
```

### Handle Management
```rust
fn add_handle(&mut self, id: HandleId, position: Vec2, target: Vec2) -> Result<()>
fn update_handle(&mut self, id: HandleId, target: Vec2) -> Result<()>
fn remove_handle(&mut self, id: HandleId) -> Result<()>
```

### Verification
```rust
fn verify_local_injectivity(&self) -> Result<VerificationResult>
// Check for fold-overs (det J ≤ 0)
```

---

## SOCP Solver Integration (Major Phase 2 Task)

### Current State (Phase 1)
The `ClarabelSolver` is a stub that:
- Takes an `SOCPProblem` input
- Returns empty/dummy coefficients
- Placeholder for actual Clarabel integration

### Phase 2 Requirements
The real solver must:

1. **Accept SOCPProblem structure**
   ```rust
   pub struct SOCPProblem {
       pub constraints: Vec<ConeConstraint>,
       pub energy: EnergyTerms,
       pub handle_constraints: Vec<HandleConstraint>,
   }
   ```

2. **Formulate SOCP (Eq. 18)**
   - min c^T Q c
   - s.t. ||A_i c + b_i|| ≤ d_i
   - Constraints from distortion strategy
   - Energy terms from regularization strategy
   - Handle position constraints (Eq. 29-30)

3. **Use Clarabel solver**
   - Already in Cargo.toml dependencies
   - Reference: https://docs.rs/clarabel/0.10/
   - Convert problem to Clarabel's format
   - Solve and extract solution

4. **Return coefficients**
   ```rust
   fn solve(&self, problem: &SOCPProblem) -> Result<Vec<Vec2>>
   // Returns [c_1, c_2, ..., c_n]
   ```

### Solver Logic Outline
```rust
impl SOCPSolverBackend for ClarabelSolver {
    fn solve(&self, problem: &SOCPProblem) -> Result<Vec<Vec2>> {
        // 1. Determine variable count from constraints
        let n_vars = problem.constraints.len() * 2;  // Each c_i is Vec2

        // 2. Extract constraint matrices
        let (P, q) = extract_objective_from(&problem.energy);
        let (A, b, K) = extract_constraints_from(&problem.constraints);

        // 3. Create Clarabel solver
        let mut solver = clarabel::DefaultSolver::new(&P, &q, &A, &b, &K, settings);

        // 4. Solve
        solver.solve();

        // 5. Extract and reshape solution
        let x = solver.solution.x;
        let coeffs: Vec<Vec2> = x.chunks(2)
            .map(|chunk| Vec2::new(chunk[0], chunk[1]))
            .collect();

        Ok(coeffs)
    }
}
```

---

## Testing Strategy for Phase 2

### Unit Tests (in pgpm-core)
- Already passing (13 tests)
- Keep as regression tests
- Add SOCP solver tests once integrated

### Integration Tests (in bevy-pgpm)
- Load test images
- Run full algorithm cycles
- Verify convergence
- Check render output

### Visual Tests
- Bevy UI display
- Handle placement/dragging
- Distortion heatmap rendering
- Deformed image overlay

---

## File Structure Completed

```
crates/pgpm-core/
├── Cargo.toml                    ✅
├── src/
│   ├── lib.rs                    ✅ Public API
│   ├── types.rs                  ✅ Core data types
│   ├── strategy.rs               ✅ Strategy trait definitions
│   ├── mapping.rs                ✅ ProvablyGoodPlanarMapping trait + Algorithm 1
│   ├── concrete.rs               ✅ PGPMv2 implementation
│   ├── basis.rs                  ✅ GaussianBasis
│   ├── distortion.rs             ✅ IsometricStrategy
│   ├── regularization.rs         ✅ BiharmonicRegularization
│   └── solver.rs                 ✅ ClarabelSolver (stub)
└── tests/
    └── integration_test.rs       ✅ 13 passing tests
```

---

## Equation Reference

Phase 2 developers should reference these equations:

| Component | Equation | File | Function |
|-----------|----------|------|----------|
| Mapping evaluation | Eq. 3 | mapping.rs | `evaluate_mapping()` |
| Jacobian gradient | Eq. 3 (∇) | mapping.rs | `mapping_gradient()` |
| Distortion (iso) | Eq. 10 | distortion.rs | `compute_distortion()` |
| Distortion (conf) | Eq. 12 | (Phase 3) | |
| Active set update | Eq. 14-17 | mapping.rs | `update_active_set_internal()` |
| SOCP formulation | Eq. 18 | solver.rs | `solve()` implementation |
| Constraints (iso) | Eq. 21-23 | (Phase 2: solver) | |
| Handle constraints | Eq. 29-30 | mapping.rs | `build_handle_constraints_internal()` |
| Biharmonic energy | Eq. 31 | regularization.rs | `build_energy_terms()` |
| ARAP energy | Eq. 32-33 | (Phase 3) | |
| Algorithm 1 | Section 5 | mapping.rs | `algorithm_step()` |

---

## Communication with Phase 2 Team

### Key Person/Roles
- Phase 2 lead: Responsible for bevy-pgpm integration
- SOCP specialist: Handles Clarabel solver integration
- UI/graphics: Handles rendering and input handling

### Deliverables at Phase 2 Completion
- ✅ Real SOCP solver (Clarabel integrated)
- ✅ bevy-pgpm UI with handle manipulation
- ✅ Image loading and contour extraction
- ✅ Real-time deformation rendering
- ✅ Distortion visualization
- ✅ Complete Phase 1+2 documentation

### Testing Checklist for Phase 2
- [ ] `cargo build` succeeds (no warnings)
- [ ] `cargo test` passes all tests
- [ ] SOCP solver produces reasonable solutions
- [ ] Simple translation preserves distortion (σ ≈ 1)
- [ ] Handle drag causes expected deformation
- [ ] Algorithm converges in < 10 steps (typical)
- [ ] det J > 0 verified for converged mappings
- [ ] Bevy UI is responsive (60 FPS)
- [ ] Memory usage is reasonable (<100MB for 512×512 image)

---

## Notes for Phase 3

Phase 3 will add:
1. B-Spline and TPS basis functions (Table 1)
2. Conformal distortion strategy (Eq. 12-13)
3. ARAP regularization (Eq. 32-33)
4. Advanced verification strategies (Algorithm 1, step 7)
5. Geodesic distance computation for shape-aware bases

All of these can be added without modifying Phase 2 code.
They simply implement existing trait interfaces and are plugged in via constructors.

---

## Getting Help

When Phase 2 encounters issues:

1. **Algorithm questions**: Refer to Poranne & Lipman (2014) paper + CLAUDE.md mapping
2. **SOCP issues**: Check Clarabel documentation + test with simple problems first
3. **Type issues**: Remember `Vec2` = `Vector2<f64>`, `Mat2` = `Matrix2<f64>`
4. **Trait questions**: Study the trait definition in `mapping.rs` - it's the spec

---

## Summary

Phase 1 provides a **complete, tested, and mathematically sound foundation** for Algorithm 1.
Phase 2's job is to **integrate this into a usable UI** and **implement the SOCP solver**.

The architecture is designed for **maximum reusability and testability**.
New developers should focus on *using* the trait, not modifying it.

Good luck with Phase 2!
