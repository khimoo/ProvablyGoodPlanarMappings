# Provably Good Planar Mappings Refactoring Plan (v2)

このドキュメントは、現状のPython実装 (`v2/bevy_image_deform/scripts/`) を論文 "Provably Good Planar Mappings" (Poranne & Lipman, 2014) の理論的保証に忠実な形へリファクタリングするための計画書です。

## 1. 現状の問題点と課題

### 1.1. 理論的保証 (Injectivity Guarantee) の欠如
論文の核心は「Collocation Points (選点) で歪みを $K$ 以下に抑えれば、グリッド密度 $h$ に応じて領域全体でも歪みが $K_{\max}$ 以下に収まり、単射性 (Injectivity) が保証される」という点です。
*   **現状:** $h$ (グリッド間隔) と $K, K_{\max}$ の関係式、および Modulus of Continuity $\omega(t)$ の計算が実装されていません。
*   **リスク:** 頂点密度が不足している箇所で裏返りが発生する可能性や、過剰な計算によるパフォーマンス低下。

### 1.2. 評価点と描画点の混同
*   **あるべき姿:**
    *   **Collocation Points ($Z$):** 歪みを抑制するために理論的に導出された密度を持つグリッド点。
    *   **Vertices:** 実際にテクスチャマッピングされ描画されるメッシュ頂点。
*   **現状:** 入力された `vertices` をそのまま歪み評価点として使用しており、ソルバーの制約条件と描画対象が混ざっています。

### 1.3. Active Set 戦略と更新ロジック
*   **現状:** Active Set の更新基準が単純な閾値比較のみです。論文で言及されている Active Set 外の点に対する厳密なバウンドチェック (Eq. 11, 13) が欠けています。
*   **Frame ($d_i$) の管理:** 更新タイミングが不明瞭で、ソルバー内部で都度計算するなど冗長な部分があります。

### 1.4. クラス設計の複雑性 (Dense Coupling)
*   **継承構造:** `BetterFitwithGaussianRBF` が `ProvablyGoodPlanarMapping` と `RBF2DandAffine` を多重継承しており、初期化や責務が分散しています。
*   **パラメータ:** $\epsilon$ や基底関数の配置戦略がハードコードされており、調整が困難です。

---

## 2. リファクタリングの目標

1.  **理論保証モードの実装:** $K$ と $K_{\max}$ から必要なグリッド密度 $h$ を逆算し、厳密な単射性を保証するモードをデフォルトとする。
2.  **責務の分離:** 「アルゴリズムのフロー (Algorithm 1)」と「基底関数の計算 (Gaussian RBF等)」を分離する。
3.  **データ構造の整理:** ソルバー用のグリッド ($Z$) と描画用の頂点を明確に区別する。

---

## 3. アーキテクチャ設計

### 3.1. 歪み制御戦略設定 (Configuration & Strategy)

アルゴリズムの制御パラメータを `SolverConfig` に集約し、歪み制御戦略 (Distortion Strategy) は Union 型等で表現します。これにより、初期化パラメータの依存関係を型レベルで明確にします。

```python
@dataclass
class SolverConfig:
    domain_bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    source_handles: np.ndarray  # (N, 2) Init control points
    basis_type: str = "Gaussian"  # "Gaussian" | "TPS" | "B-Spline"
    strategy: 'DistortionStrategyConfig'  # 具体的な戦略設定

class DistortionStrategyConfig:
    pass

@dataclass
class FixedBoundCalcGrid(DistortionStrategyConfig):
    """Strategy 2 (Default): 品質保証モード。K_maxを守るために必要なhを自動計算。"""
    K: float = 2.0      # Target distortion on grid
    K_max: float = 10.0 # Guaranteed max distortion

@dataclass
class FixedGridCalcBound(DistortionStrategyConfig):
    """Strategy 1: グリッド固定モード。hを固定し、理論上のK_maxを計算(許容)。"""
    grid_resolution: Tuple[int, int]
    K: float = 2.0 

@dataclass
class HeuristicFast(DistortionStrategyConfig):
    """Strategy 3: 高速モード。厳密な保証チェックをスキップ。"""
    grid_resolution: Tuple[int, int]
```

### 3.2. 基底クラス `ProvablyGoodPlanarMapping`

Algorithm 1 のフロー制御を行うメインクラス。状態を持ち、入力された設定に基づいて初期化されます。

**責務:**
*   構成(`SolverConfig`)に基づく **Collocation Grid ($Z$) の自動生成と管理**
*   ユーザーからの入力(`dst_handles`)を受け取り、係数($c$)を更新
*   任意の点群(`points`)に対する変形適用(`transform`)

```python
class ProvablyGoodPlanarMapping(metaclass=ABCMeta):
    def __init__(self, config: SolverConfig):
        self.config = config
        self.collocation_grid: np.ndarray = None # Z: (M, 2)
        # ...他ソルバー内部状態...
        self._initialize_solver()

    def _initialize_solver(self):
        """
        Configに基づき h を決定し、Collocation Grid Z を生成する。
        ユーザーの描画メッシュとは独立して決定される。
        """
        pass

    def compute_mapping(self, target_handles: np.ndarray) -> None:
        """
        Algorithm 1: Main Loop Phase
        制御点の移動を受け取り、最適化を実行して写像係数 c を更新する。
        
        Args:
            target_handles: (N, 2) Current positions of control points
        """
        pass

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        現在の係数 c を用いて、任意の点群を変形する。
        
        Args:
           points: (K, 2) Arbitrary points (e.g. render mesh vertices)
        Returns:
           (K, 2) Deformed points
        """
        pass

    # ...内部メソッド (update_active_set など)...
```

### 3.3. 基底関数の抽象化 `BasisFunction`

数値計算部分を切り出します。これにより `RBF2DandAffine` クラスは廃止またはこのインターフェースの実装として再利用します。

```python
class BasisFunction(metaclass=ABCMeta):
    @abstractmethod
    def set_centers(self, centers: np.ndarray) -> None:
        """RBFの中心点などを設定"""
        pass
    
    @abstractmethod
    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        """
        基底関数の値を計算 (\Phi 行列)
        Returns: (M, N+3) Matrix
        """
        pass
    
    @abstractmethod
    def jacobian(self, coords: np.ndarray) -> np.ndarray:
        """
        基底関数の偏微分 (Jacobian) を計算
        Returns: (M, 2, N+3) Tensor or similar structure
        """
        pass
    
    @abstractmethod
    def compute_h_from_distortion(self, K: float, K_max: float) -> float:
        """K_max >= K + omega(h) を満たす h を逆算する"""
        pass
    
    @abstractmethod
    def get_identity_coefficients(self, src_handles: np.ndarray) -> np.ndarray:
        """
        恒等写像となる初期係数を返す
        Returns: (N+3, 2) Coefficients
        """
        pass
```

### 3.4. 実装クラス `GaussianRBF` & `BetterFitwithGaussianRBF`

`ProvablyGoodPlanarMapping` を継承し、Gaussian RBF に特化した実装を行います。

**GaussianRBF (BasisFunctionの実装):**
*   **パラメータ同値性の保証:**
    *   現行実装: $\phi(r) = \exp(-(r/\epsilon)^2)$
    *   論文表記: $\phi(r) = \exp(-r^2/(2s^2))$
    *   変換: $\epsilon = s\sqrt{2}$ (または $s = \epsilon/\sqrt{2}$) として内部で統一的に扱う。これにより論文の $\omega(t) = t/s^2$ の $s$ を正しく導出する。
*   $\omega(t) \approx \frac{t}{s^2}$ (論文 Table 1) などの Modulus of Continuity を実装。
*   $s$ (Gaussian width) と $h$ (Grid spacing) の関係を管理 (例: $s = \alpha h$)。

**BetterFitwithGaussianRBF:**
*   `src_handles` (制御点) を RBF の中心として設定。
*   `compute_h_from_distortion` を用いて、保証に必要なグリッド解像度を決定。

---

## 4. データフローと入出力設計 (Data I/O Strategy)

アルゴリズムは「設定に基づく状態を持つ変換機」として振る舞います。ユーザーは初期化時に戦略を決定し、実行時は制御点の移動のみを入力します。描画用データは任意のタイミングで変換可能です。

### 4.1. 初期化シーケンス (Initialization)

1.  **Config構築**: `SolverConfig` に領域境界、初期制御点、そして「歪み制御戦略」を設定する。
2.  **Solverインスタンス化**: `ProvablyGoodMapping(config)`
    *   Solverは内部で設定に基づいた **Collocation Grid ($Z$)** を自動生成する。
    *   ユーザーの描画用メッシュ等はここでは不要。

### 4.2. 実行時ループ (Runtime)

1.  **計算 (Compute):**
    *   `solver.compute_mapping(target_handles)`
    *   制御点の現在位置を入力。Solverは内部グリッド $Z$ 上で最適化を行い、係数 $c$ を更新。
2.  **変換 (Transform):**
    *   `solver.transform(render_mesh_vertices)`
    *   描画用の頂点配列を渡し、変形後の座標を受け取る。
    *   このメソッドはピュアな関数に近い（内部状態 $c$ を使うだけ）。

### 4.3. コード例

```python
# 0. Set up configuration
config = SolverConfig(
    domain_bounds=(0, 0, 512, 512),
    source_handles=initial_handles,
    # Strategy 2: Guarantee Injectivity (Calc h from K, K_max)
    strategy=FixedBoundCalcGrid(K=2.0, K_max=10.0) 
)

# 1. Initialize Solver (Internal grid Z is generated here)
solver = ProvablyGoodMapping(config)

# --- Game Loop ---
while running:
    # 2. Update Constraints
    current_handles = get_handle_positions()
    solver.compute_mapping(target_handles=current_handles)

    # 3. Apply Deformation (Independent of internal grid resolution)
    render_mesh.vertices = solver.transform(original_mesh.vertices)
```

---

## 5. 実装プロセス計画


### Step 1: クラス構造の再編
1.  `ProvablyGoodPlanarMapping` を抽象クラス化し、テンプレートメソッドパターンを適用。
2.  `BasisFunction` インターフェースを定義。
3.  `GaussianRBF` クラスを作成し、既存の計算ロジックを移植。

### Step 2: 理論部分の実装 ($\omega(t)$)
1.  Gaussian RBF の Lipschitz constant 導出ロジックを実装。
2.  $K, K_{\max}$ から $h$ を計算する `compute_h_from_distortion` の実装。

### Step 3: グリッド生成と分離
1.  `initialize_solver` 内で $h$ に基づくグリッド生成を実装。
2.  描画用 `vertices` とソルバー用 `collocation_grid` を完全に分離して動作するように修正。

### Step 4: Active Set 戦略の厳密化 (Optional / Phase 2)
1.  論文 Eq. 11, 13 に基づく厳密な Active Set 追加判定の実装。
---
