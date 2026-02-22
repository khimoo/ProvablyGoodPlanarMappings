# Strategy 1 用 API 設計 (修正版)

## 論文の3つのStrategy の正確な定義

### Strategy 1: Given Z and K → bound K_max
```
入力: コロケーションポイント集合 Z、コロケーション点上の歪み上限 K
出力: 領域全体での最大歪み K_max の上界
計算: 式(11)または(13)から直接計算

K_max は計算結果であり、入力ではない！
```

**式(11) isometric の場合:**
```
D_iso(x) ≤ max{ K + ω(h), 1/(1/K - ω(h)) }
```

**重要:** K_max は Z と K から自動的に決定される

---

### Strategy 2: Given K and K_max → calculate h
```
入力: コロケーション点上の歪み上限 K、領域全体での目標歪み上限 K_max
出力: 必要なコロケーションポイント密度 h
計算: 式(14)または(15)を h について逆算

h = ω^(-1)( min{ K_max - K, 1/K - 1/K_max } )
```

**重要:** K_max は入力であり、h を決定するために必要

---

### Strategy 3: Given Z and K_max → calculate K
```
入力: コロケーションポイント集合 Z、領域全体での目標歪み上限 K_max
出力: コロケーション点上で強制すべき歪み上限 K
計算: 式(16)または(17)を K について逆算

K = min{ K_max - ω(h), 1/(1/K_max + ω(h)) }
```

**重要:** K_max は入力であり、K を決定するために必要

---

## 現在の実装の問題点

### 現在の deform_algo.py

```python
class BetterFitwithGaussian(ProvablyGoodPlanarMapping):
    def __init__(self, domain_bounds, guarantee_strategy=None, 
                 s_param=1.0, K_solver=2.0, K_max=5.0):
        # K_solver と K_max が常に指定されている
        self.K_solver = K_solver
        self.K_max = K_max
```

**問題:**
- Strategy 1 では K_max は不要（計算結果）
- Strategy 2 では K_max は必須（h を計算するため）
- 両方に同じパラメータを渡すのは設計が悪い

---

## 正しい API 設計

### 1. Strategy 1: コロケーション密度と K を指定

```python
class Strategy1(GuaranteeStrategy):
    """
    Strategy 1: Given Z and K → bound K_max
    
    コロケーションポイント密度 h と K を指定し、
    結果として得られる K_max を計算する。
    K_max は出力であり、入力ではない。
    """
    
    def __init__(self, domain_bounds: Tuple[float, float, float, float],
                 collocation_resolution: int = 500,
                 K_on_collocation: float = 3.5):
        """
        Args:
            domain_bounds: (x_min, x_max, y_min, y_max)
            collocation_resolution: グリッド解像度
            K_on_collocation: コロケーション点上の歪み上限 K
        
        K_max は計算結果として得られる（入力ではない）
        """
        super().__init__(domain_bounds)
        self.collocation_resolution = collocation_resolution
        self.K_on_collocation = K_on_collocation
    
    def get_initial_h(self, basis, K_solver, K_max):
        """
        Strategy 1: h は事前に決定されている
        
        注: K_solver と K_max は渡されるが、Strategy 1 では使用しない
        （Strategy 2 との互換性のため）
        """
        x_min, x_max, y_min, y_max = self.bounds
        max_span = max(x_max - x_min, y_max - y_min)
        return float(max_span / self.collocation_resolution)
    
    def get_strict_h_after_drag(self, basis, K_solver, K_max, c):
        """Strategy 1: グリッド細分化なし"""
        return float('inf')
    
    def compute_guaranteed_K_max(self, basis, h):
        """
        Strategy 1 の出力: 実際の K_max を計算
        
        式(11): D_iso(x) ≤ max{ K + ω(h), 1/(1/K - ω(h)) }
        
        Returns:
            K_max: 領域全体で保証される最大歪み
        """
        K = self.K_on_collocation
        omega_h = 2.0 * 1.0 * basis.compute_basis_gradient_modulus(1.0) * h
        
        # 式(11)
        K_max_bound = max(
            K + omega_h,
            1.0 / (1.0 / K - omega_h)
        )
        
        return float(K_max_bound)
```

### 2. Strategy 2: K と K_max を指定

```python
class Strategy2(GuaranteeStrategy):
    """
    Strategy 2: Given K and K_max → calculate h
    
    目標 K_max を達成するために必要な h を計算する。
    K_max は入力であり、h を決定するために必須。
    """
    
    def __init__(self, domain_bounds: Tuple[float, float, float, float],
                 interactive_resolution: int = 200):
        """
        Args:
            domain_bounds: (x_min, x_max, y_min, y_max)
            interactive_resolution: ドラッグ中のグリッド解像度
        
        K_max は ProvablyGoodPlanarMapping から渡される
        """
        super().__init__(domain_bounds)
        self.interactive_resolution = interactive_resolution
    
    def get_initial_h(self, basis, K_solver, K_max):
        """
        Strategy 2: 初期 h は固定解像度から計算
        
        K_max は使用しない（ドラッグ中は粗いグリッド）
        """
        x_min, x_max, y_min, y_max = self.bounds
        max_span = max(x_max - x_min, y_max - y_min)
        return float(max_span / self.interactive_resolution)
    
    def get_strict_h_after_drag(self, basis, K_solver, K_max, c):
        """
        Strategy 2: ドラッグ後に必要な h を計算
        
        式(14): h ≤ ω^(-1)( min{ K_max - K, 1/K - 1/K_max } )
        
        K_max は必須（目標歪み）
        """
        margin = min(K_max - K_solver, (1.0 / K_solver) - (1.0 / K_max))
        
        if margin <= 0:
            warnings.warn("Margin is zero or negative. Returning fallback h.")
            return 1e-5
        
        c_rbf = self._extract_rbf_coefficients(c)
        max_coeff_norm = np.max(np.sum(np.abs(c_rbf), axis=1))
        
        if max_coeff_norm < 1e-12:
            return float('inf')
        
        C = basis.compute_basis_gradient_modulus(1.0)
        h_strict = margin / (2.0 * max_coeff_norm * C)
        return float(h_strict)
```

### 3. BevyBridge の修正

```python
class BevyBridge:
    def __init__(self, strategy_type: str = "strategy2",
                 strategy_params: Optional[Dict] = None):
        """
        Args:
            strategy_type: "strategy1" or "strategy2"
            strategy_params:
                - strategy1:
                    - collocation_resolution: int
                    - K_on_collocation: float (コロケーション点上の K)
                    # K_max は計算結果（指定しない）
                - strategy2:
                    - interactive_resolution: int
                    - K_solver: float (コロケーション点上の K)
                    - K_max: float (目標 K_max）
        """
        self.strategy_type = strategy_type
        self.strategy_params = strategy_params or {}
        self.solver: Optional[BetterFitwithGaussian] = None
        self.image_width: float = 0.0
        self.image_height: float = 0.0
        self.contour: Optional[np.ndarray] = None
        self.control_points: List[Tuple[int, float, float]] = []
        self.is_setup_finalized: bool = False
        self.guaranteed_K_max: Optional[float] = None  # Strategy 1 の出力
    
    def initialize_domain(self, image_width: float, image_height: float,
                         epsilon: float) -> None:
        """Initialize the deformation domain with specified strategy."""
        self.image_width = image_width
        self.image_height = image_height
        
        domain_bounds = (0.0, image_width, 0.0, image_height)
        
        if self.strategy_type == "strategy1":
            strategy, K_solver, K_max = self._create_strategy1(domain_bounds)
        else:  # strategy2
            strategy, K_solver, K_max = self._create_strategy2(domain_bounds)
        
        self.solver = BetterFitwithGaussian(
            domain_bounds=domain_bounds,
            guarantee_strategy=strategy,
            s_param=epsilon,
            K_solver=K_solver,
            K_max=K_max
        )
        
        print(f"Initialized domain: {image_width}x{image_height}, epsilon={epsilon}")
        print(f"Strategy: {self.strategy_type}")
        if self.strategy_type == "strategy1":
            print(f"  K_on_collocation: {K_solver}")
            print(f"  (K_max will be computed after finalize_setup)")
        else:
            print(f"  K_solver: {K_solver}, K_max: {K_max}")
    
    def _create_strategy1(self, domain_bounds):
        """
        Create Strategy 1: Given Z and K → bound K_max
        
        Returns:
            (strategy, K_solver, K_max)
            K_max は Strategy 2 との互換性のため渡すが、Strategy 1 では使用しない
        """
        from deform_algo import Strategy1
        
        collocation_resolution = self.strategy_params.get(
            'collocation_resolution', 500
        )
        K_on_collocation = self.strategy_params.get(
            'K_on_collocation', 3.5
        )
        
        strategy = Strategy1(
            domain_bounds=domain_bounds,
            collocation_resolution=collocation_resolution,
            K_on_collocation=K_on_collocation
        )
        
        # Strategy 1 では K_max は計算結果なので、ダミー値を渡す
        # （Strategy 2 との互換性のため）
        K_max_dummy = 10.0  # 実際の値は finalize_setup() で計算
        
        return strategy, K_on_collocation, K_max_dummy
    
    def _create_strategy2(self, domain_bounds):
        """
        Create Strategy 2: Given K and K_max → calculate h
        
        Returns:
            (strategy, K_solver, K_max)
            K_max は必須（h を計算するため）
        """
        from deform_algo import Strategy2
        
        interactive_resolution = self.strategy_params.get(
            'interactive_resolution', 200
        )
        K_solver = self.strategy_params.get('K_solver', 3.5)
        K_max = self.strategy_params.get('K_max', 5.0)
        
        strategy = Strategy2(
            domain_bounds=domain_bounds,
            interactive_resolution=interactive_resolution
        )
        
        return strategy, K_solver, K_max
    
    def finalize_setup(self) -> None:
        """
        Finalize setup phase and initialize the solver.
        
        Strategy 1 の場合、ここで K_max を計算して記録する。
        """
        if self.is_setup_finalized:
            return
        
        if len(self.control_points) == 0:
            raise RuntimeError("No control points added")
        
        if self.solver is None:
            raise RuntimeError("Solver not initialized")
        
        src_handles = np.array(
            [[x, y] for (_, x, y) in self.control_points],
            dtype=np.float64
        )
        
        self.solver.initialize_mapping(src_handles)
        
        # Strategy 1 の場合、K_max を計算
        if self.strategy_type == "strategy1":
            h = self.solver.current_h
            strategy = self.solver.strategy
            self.guaranteed_K_max = strategy.compute_guaranteed_K_max(
                self.solver.basis, h
            )
            print(f"Strategy 1: Guaranteed K_max = {self.guaranteed_K_max:.4f}")
        
        collocation_count = len(self.solver.collocation_points)
        print(f"Generated {collocation_count} collocation points")
        
        if self.contour is not None:
            original_count = len(self.solver.collocation_points)
            self.solver.collocation_points = filter_points_inside_contour(
                self.solver.collocation_points,
                self.contour
            )
            filtered_count = len(self.solver.collocation_points)
            print(f"Filtered collocation points: {original_count} -> {filtered_count}")
            
            self.solver._update_hessian_term()
        
        self.is_setup_finalized = True
        print(f"Finalized setup with {len(self.control_points)} control points")
    
    # 他のメソッドは変更なし
    def set_contour(self, contour_points: List[Tuple[float, float]]) -> None:
        self.contour = np.array(contour_points, dtype=np.float64)
        print(f"Set contour with {len(contour_points)} points")
    
    def add_control_point(self, control_index: int, x: float, y: float) -> None:
        if self.is_setup_finalized:
            raise RuntimeError("Cannot add control points after finalize_setup")
        self.control_points.append((control_index, x, y))
        print(f"Added control point {control_index} at ({x:.1f}, {y:.1f})")
    
    def start_drag_operation(self) -> None:
        if not self.is_setup_finalized:
            raise RuntimeError("Setup not finalized")
        if self.solver is None:
            return
        src_handles = np.array(
            [[x, y] for (_, x, y) in self.control_points],
            dtype=np.float64
        )
        self.solver.start_drag(src_handles)
        print("Started drag operation")
    
    def update_control_point(self, control_index: int, x: float, y: float) -> None:
        if not self.is_setup_finalized:
            raise RuntimeError("Setup not finalized")
        if control_index < 0 or control_index >= len(self.control_points):
            raise ValueError(f"Invalid control_index: {control_index}")
        idx, _, _ = self.control_points[control_index]
        self.control_points[control_index] = (idx, x, y)
    
    def solve_frame(self, inverse_grid_resolution: int = 64) -> None:
        if not self.is_setup_finalized:
            raise RuntimeError("Setup not finalized")
        if self.solver is None:
            raise RuntimeError("Solver not initialized")
        target_handles = np.array(
            [[x, y] for (_, x, y) in self.control_points],
            dtype=np.float64
        )
        self.solver.update_drag(target_handles, num_iterations=2)
    
    def get_basis_parameters(self) -> Dict:
        if self.solver is None:
            raise RuntimeError("Solver not initialized")
        return {
            'coefficients': self.solver.coefficients.tolist(),
            'centers': self.solver.basis.centers.tolist(),
            's_param': float(self.solver.basis.s),
            'n_rbf': len(self.control_points),
        }
    
    def end_drag_operation(self) -> None:
        if not self.is_setup_finalized:
            raise RuntimeError("Setup not finalized")
        if self.solver is None:
            return
        target_handles = np.array(
            [[x, y] for (_, x, y) in self.control_points],
            dtype=np.float64
        )
        was_refined = self.solver.end_drag(target_handles)
        if was_refined:
            print(f"Grid was refined by {self.strategy_type}")
            if self.contour is not None:
                self.solver.collocation_points = filter_points_inside_contour(
                    self.solver.collocation_points,
                    self.contour
                )
                self.solver._update_hessian_term()
    
    def reset_mesh(self) -> None:
        self.control_points.clear()
        self.is_setup_finalized = False
        self.solver = None
        self.contour = None
        self.guaranteed_K_max = None
        print("Reset mesh")
```

---

## 使用例

### Strategy 1: コロケーション密度と K を指定

```python
# K_on_collocation = 3.5 で固定
# K_max は計算結果として得られる
bridge = BevyBridge(
    strategy_type="strategy1",
    strategy_params={
        'collocation_resolution': 500,
        'K_on_collocation': 3.5
        # K_max は指定しない（計算結果）
    }
)

bridge.initialize_domain(800.0, 800.0, 40.0)
bridge.add_control_point(0, 100.0, 100.0)
bridge.add_control_point(1, 700.0, 700.0)
bridge.finalize_setup()

# 出力例:
# Strategy 1: Guaranteed K_max = 4.2
```

### Strategy 2: K と K_max を指定

```python
# K_solver = 3.5, K_max = 5.0 で指定
# h は計算結果として得られる
bridge = BevyBridge(
    strategy_type="strategy2",
    strategy_params={
        'interactive_resolution': 200,
        'K_solver': 3.5,
        'K_max': 5.0  # 必須
    }
)

bridge.initialize_domain(800.0, 800.0, 40.0)
bridge.add_control_point(0, 100.0, 100.0)
bridge.add_control_point(1, 700.0, 700.0)
bridge.finalize_setup()

# ドラッグ終了時に必要に応じて h が細分化される
```

---

## API 設計の改善点

### 1. Strategy 1 と Strategy 2 の入出力が明確

| Strategy | 入力 | 出力 |
|----------|------|------|
| Strategy 1 | Z, K | K_max |
| Strategy 2 | K, K_max | h |

### 2. K_max の役割が明確

```
Strategy 1:
  - K_max は指定しない
  - 計算結果として guaranteed_K_max が得られる
  - finalize_setup() 後に確認可能

Strategy 2:
  - K_max は必須パラメータ
  - h を計算するために使用
  - ドラッグ終了時の細分化判定に使用
```

### 3. パラメータの一貫性

```python
# Strategy 1
strategy_params = {
    'collocation_resolution': 500,
    'K_on_collocation': 3.5
}

# Strategy 2
strategy_params = {
    'interactive_resolution': 200,
    'K_solver': 3.5,
    'K_max': 5.0
}
```

---

## 実装チェックリスト

- [ ] `Strategy1.compute_guaranteed_K_max()` を実装
- [ ] `BevyBridge.__init__()` に `guaranteed_K_max` を追加
- [ ] `_create_strategy1()` で K_max をダミー値で渡す
- [ ] `_create_strategy2()` で K_max を必須パラメータとする
- [ ] `finalize_setup()` で Strategy 1 の K_max を計算
- [ ] ログ出力で Strategy 1 の K_max を表示
- [ ] テスト: Strategy 1 と Strategy 2 の K_max の違いを確認

---

## まとめ

**修正前の問題:**
- Strategy 1 でも K_max を指定していた
- K_max の役割が不明確
- 論文の3つの Strategy との対応が曖昧

**修正後の改善:**
- Strategy 1: K_on_collocation を指定 → K_max が計算結果
- Strategy 2: K_solver と K_max を指定 → h が計算結果
- 論文の定義に正確に対応
- API が直感的で理解しやすい
