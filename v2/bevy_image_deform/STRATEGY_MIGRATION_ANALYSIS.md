# Strategy 1 Migration Analysis

## Current State (Strategy 2)

### bevy_bridge.py
```python
# Line 103-110: Current initialization uses Strategy 2 (default)
self.solver = BetterFitwithGaussian(
    domain_bounds=domain_bounds,
    s_param=epsilon,
    K_solver=3.5,
    K_max=5.0
)
# guarantee_strategy parameter is NOT passed, so defaults to Strategy2
```

### deform_algo.py
```python
# Line 350-356: BetterFitwithGaussian.__init__
def __init__(self, domain_bounds, guarantee_strategy=None, s_param=1.0, K_solver=2.0, K_max=5.0):
    basis = GaussianRBF(s=s_param)
    if guarantee_strategy is None:
        guarantee_strategy = Strategy2(domain_bounds)  # ← Default is Strategy2
    super().__init__(basis=basis, guarantee_strategy=guarantee_strategy, ...)
```

## Key Differences: Strategy 1 vs Strategy 2

| Aspect | Strategy 1 | Strategy 2 |
|--------|-----------|-----------|
| **Grid Computation** | Pessimistic (upfront) | Optimistic (interactive) |
| **Initial h** | Very fine (strict) | Coarse (fast) |
| **Drag Performance** | Slower (fine grid) | Faster (coarse grid) |
| **end_drag() Behavior** | No refinement (returns inf) | May refine grid |
| **Parameter** | `max_expected_c_norm` | `interactive_resolution` |
| **Use Case** | Guaranteed safety from start | Interactive responsiveness |

## Migration Strategy

### Option A: Simple Default Change (Minimal)
**Change only the default in `BetterFitwithGaussian.__init__`**

```python
# deform_algo.py line 350-356
if guarantee_strategy is None:
    guarantee_strategy = Strategy1(domain_bounds, max_expected_c_norm=100.0)  # ← Change this
```

**Pros:**
- Minimal code change (1 line)
- No changes needed to bevy_bridge.py
- Backward compatible (can still pass Strategy2 explicitly)

**Cons:**
- Hard-coded `max_expected_c_norm=100.0` may not be optimal
- No way to tune Strategy 1 parameter from Rust side

---

### Option B: Parameterized Strategy Selection (Recommended)
**Add strategy selection to BevyBridge and pass it through**

#### Changes to bevy_bridge.py:
```python
# Add import
from deform_algo import BetterFitwithGaussian, Strategy1, Strategy2

class BevyBridge:
    def __init__(self, strategy_type: str = "strategy1", max_expected_c_norm: float = 100.0):
        """
        Args:
            strategy_type: "strategy1" or "strategy2"
            max_expected_c_norm: For Strategy1, the worst-case coefficient norm estimate
        """
        self.strategy_type = strategy_type
        self.max_expected_c_norm = max_expected_c_norm
        self.solver = None
        # ... rest of init ...

    def initialize_domain(self, image_width, image_height, epsilon):
        domain_bounds = (0.0, image_width, 0.0, image_height)
        
        # Create appropriate strategy
        if self.strategy_type == "strategy1":
            strategy = Strategy1(domain_bounds, max_expected_c_norm=self.max_expected_c_norm)
        else:
            strategy = Strategy2(domain_bounds, interactive_resolution=200)
        
        self.solver = BetterFitwithGaussian(
            domain_bounds=domain_bounds,
            guarantee_strategy=strategy,  # ← Pass strategy explicitly
            s_param=epsilon,
            K_solver=3.5,
            K_max=5.0
        )
```

#### Changes to bevy_bridge.rs:
```rust
// Add new command variant
pub enum PyCommand {
    InitializeDomain { width: f32, height: f32, epsilon: f32, strategy: String },
    // ... rest of variants ...
}

// In python_thread_loop, when creating BevyBridge:
let bridge_instance = bridge_class.call1((strategy_type,))?;
// Or pass as kwargs if BevyBridge.__init__ supports it
```

#### Changes to main.rs:
```rust
// In setup() function, pass strategy parameter
let _ = tx_cmd.try_send(PyCommand::InitializeDomain {
    width: image_width,
    height: image_height,
    epsilon,
    strategy: "strategy1".to_string(),  // ← Add this
});
```

**Pros:**
- Flexible: can switch strategies at runtime
- Tunable: can adjust `max_expected_c_norm` per use case
- Clean separation of concerns

**Cons:**
- More code changes across 3 files
- Requires Rust-Python interface changes

---

### Option C: Environment Variable / Config File
**Use external configuration to select strategy**

```python
# bevy_bridge.py
import os

class BevyBridge:
    def __init__(self):
        strategy_type = os.getenv("DEFORM_STRATEGY", "strategy1")
        max_norm = float(os.getenv("DEFORM_MAX_NORM", "100.0"))
        
        if strategy_type == "strategy1":
            self.strategy = Strategy1(domain_bounds, max_expected_c_norm=max_norm)
        else:
            self.strategy = Strategy2(domain_bounds)
```

**Pros:**
- No code changes needed in Rust
- Easy to switch without recompilation

**Cons:**
- Less discoverable
- Requires environment setup

---

## Recommended Approach: Option B

### Why Option B?
1. **Explicit and clear**: Strategy choice is visible in code
2. **Flexible**: Can tune `max_expected_c_norm` per image/use case
3. **Future-proof**: Easy to add more strategies later
4. **Testable**: Can test both strategies in same codebase

### Implementation Steps

#### Step 1: Update bevy_bridge.py
- Add `strategy_type` parameter to `BevyBridge.__init__`
- Add `max_expected_c_norm` parameter
- Create appropriate strategy in `initialize_domain()`
- Pass strategy to `BetterFitwithGaussian`

#### Step 2: Update PyCommand enum in bridge.rs
- Add `strategy: String` field to `InitializeDomain` variant
- Update the Python call site to extract and pass strategy

#### Step 3: Update main.rs
- Pass strategy name when sending `InitializeDomain` command
- Could make it configurable via command-line arg or constant

#### Step 4: Update bevy_bridge.rs
- Extract strategy parameter from command
- Pass to Python's `initialize_domain()` method

---

## Performance Implications

### Strategy 1 (Pessimistic)
- **Initialization**: Slower (computes very fine grid upfront)
- **Drag**: Slower (fine grid = more collocation points = more computation)
- **End Drag**: Instant (no refinement needed)
- **Total**: Slower overall, but guaranteed safety from start

### Strategy 2 (Optimistic)
- **Initialization**: Faster (coarse grid)
- **Drag**: Faster (coarse grid = fewer collocation points)
- **End Drag**: May be slow (if grid refinement needed)
- **Total**: Faster for typical interactions, but may spike at end

### For this project:
- **Strategy 1** is better if: You want guaranteed distortion bounds from the start
- **Strategy 2** is better if: You prioritize interactive responsiveness

---

## Testing Considerations

### What to verify after migration:
1. **Grid generation**: Strategy 1 should produce finer grid than Strategy 2
2. **Drag performance**: Strategy 1 should be slower during drag
3. **End drag**: Strategy 1 should never refine (returns inf)
4. **Distortion bounds**: Both should maintain K_max constraint
5. **Visual quality**: Deformation should look similar, just with different performance profile

### Test cases:
```python
# Test Strategy 1 initialization
strategy1 = Strategy1((0, 100, 0, 100), max_expected_c_norm=100.0)
h1 = strategy1.get_initial_h(basis, K_solver=3.5, K_max=5.0)

# Test Strategy 2 initialization
strategy2 = Strategy2((0, 100, 0, 100), interactive_resolution=200)
h2 = strategy2.get_initial_h(basis, K_solver=3.5, K_max=5.0)

# h1 should be much smaller than h2
assert h1 < h2, f"Strategy1 h={h1} should be < Strategy2 h={h2}"

# Test end_drag behavior
strict_h1 = strategy1.get_strict_h_after_drag(basis, K_solver, K_max, coefficients)
assert strict_h1 == float('inf'), "Strategy1 should never refine"
```

---

## Summary

**Recommended: Option B (Parameterized Strategy Selection)**

### Files to modify:
1. **bevy_bridge.py**: Add strategy parameter to `__init__` and `initialize_domain()`
2. **bridge.rs**: Add strategy field to `PyCommand::InitializeDomain`
3. **main.rs**: Pass strategy name when initializing domain

### Key changes:
- `BevyBridge.__init__(strategy_type="strategy1", max_expected_c_norm=100.0)`
- `initialize_domain()` creates appropriate strategy before creating solver
- Rust passes strategy name through command channel

### Backward compatibility:
- Default to Strategy 1 (pessimistic/safe)
- Can still use Strategy 2 by passing `strategy_type="strategy2"`
- No breaking changes to existing code
