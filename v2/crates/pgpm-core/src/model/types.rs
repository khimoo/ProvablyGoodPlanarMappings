//! PGPMアルゴリズムの共通型定義。
//!
//! 全ての型は論文の表記に直接対応する:
//! - Eq. 3: 係数行列 c ∈ R^{2×n}
//! - Section 3: 歪みの種類
//! - Algorithm 1: アルゴリズム状態

use crate::basis::BasisFunction;
use crate::model::domain::Domain;
use crate::policy::DistortionPolicy;
use nalgebra::{DMatrix, Vector2};

// ───────────────────────────────────────────────────────────────
// Eq. 3: 係数行列 c ∈ R^{2×n}
// c = [c_1, c_2, ..., c_n], c_i = (c¹_i, c²_i)^T
// (2, n) 形状の DMatrix<f64> として格納する。
// Row 0 → c¹ 係数 (u成分)
// Row 1 → c² 係数 (v成分)
// ───────────────────────────────────────────────────────────────
pub type CoefficientMatrix = DMatrix<f64>;

/// Eq. 5: ドメイン Ω のバウンディングボックス。
#[derive(Debug, Clone)]
pub struct DomainBounds {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

/// Section 5.4: 正則化エネルギーの種類。
#[derive(Debug, Clone)]
pub enum RegularizationType {
    /// E_bh のみ (Eq. 31)
    Biharmonic,
    /// E_arap のみ (Eq. 33)
    Arap,
    /// E_pos + λ_bh * E_bh + λ_arap * E_arap
    /// 論文 Section 6: Figure 5 は E_pos + 10^{-2} E_arap、
    ///                  Figure 8 は E_pos + 10^{-1} E_bh を使用
    Mixed { lambda_bh: f64, lambda_arap: f64 },
}

/// SOCPソルバーの数値調整パラメータ。
///
/// ソルバーの数値的挙動とリソース制限を制御する。
/// これらは**実装固有**のパラメータであり、論文の定式化には含まれない。
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// 係数変数（最初の2nエントリ）に対する P 行列の対角正則化。
    /// 正則化重み λ が非常に小さい場合のKKTシステムの
    /// 特異近傍を防止する。
    pub p_reg_coefficient: f64,
    /// 補助変数 (r, t, s) に対する対角正則化。
    pub p_reg_auxiliary: f64,
    /// Strategy 2 細分化の最大グリッド解像度（辺あたりの点数）。
    /// 論文 Section 6 では最大 6000² を使用。利用可能なメモリと
    /// 時間に応じて設定する。
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

/// Algorithm 1 (Section 5) の不変コンテキスト。
///
/// 単一のアルゴリズムステップ内で変化しない全データへの参照を束ねる。
/// `&mut AlgorithmState` と共に [`PlanarMapping::parts_mut`] から返され、
/// トレイトのデフォルトメソッド内での借用分割を可能にする。
pub struct MappingContext<'a> {
    /// 基底関数 φ_i (Table 1)
    pub basis: &'a dyn BasisFunction,
    /// 歪みポリシー（等長 Eq. 23/26 または等角 Eq. 28）
    pub policy: &'a dyn DistortionPolicy,
    /// アルゴリズムパラメータ（K, λ, 正則化の種類）
    pub params: &'a MappingParams,
    /// ソースハンドル位置 {p_l} (Eq. 29) — 固定
    pub source_handles: &'a [Vector2<f64>],
    /// ドメイン Ω のバウンディングボックス (Eq. 5)
    pub domain_bounds: &'a DomainBounds,
    /// ドメイン Ω（"x ∈ Ω" 判定用、Section 4）。
    /// `None` の場合はバウンディングボックス全体がドメインとなる。
    pub domain: Option<&'a dyn Domain>,
    /// SOCPソルバーの数値調整（論文由来ではない）
    pub solver_config: &'a SolverConfig,
}

/// Algorithm 1 の内部状態。
pub struct AlgorithmState {
    /// Eq. 3: 係数行列 c ∈ R^{2×n}
    pub coefficients: CoefficientMatrix,

    /// コロケーション点 Z = {z_j} (Eq. 4)、密なグリッド上にサンプリング
    pub collocation_points: Vec<Vector2<f64>>,

    /// アクティブセット Z' ⊂ Z (Algorithm 1)、collocation_points へのインデックスとして格納
    pub active_set: Vec<usize>,

    /// 安定化セット Z'' (Algorithm 1: "farthest point samples")、
    /// collocation_points へのインデックスとして格納
    pub stable_set: Vec<usize>,

    /// フレームベクトル d_i (Eq. 27)、コロケーション点ごとに1つ
    pub frames: Vec<Vector2<f64>>,

    /// K_high, K_low (Section 5 "Activation of constraints")
    /// デフォルト: K_high = 0.1 + 0.9*K, K_low = 0.5 + 0.5*K
    pub k_high: f64,
    pub k_low: f64,

    /// ドメイン内部マスク: コロケーション点がドメイン Ω（輪郭ポリゴン）の
    /// 内部にある場合に `true`。ドメイン外の点は矩形グリッド内に保持される
    /// （グリッド上の局所最大検出が引き続き機能するように）が、
    /// アクティブ/安定化セットには追加されず、ARAP正則化サンプリングからも除外される。
    ///
    /// 論文 Section 4: "consider all the points from a surrounding uniform
    /// grid that fall inside the domain"
    pub domain_mask: Vec<bool>,

    /// コロケーション点での基底関数評価の事前計算データ
    pub precomputed: Option<PrecomputedData>,

    /// 局所最大探索用のグリッド次元 (Section 5)
    pub grid_width: usize,
    pub grid_height: usize,

    /// 前ステップのターゲットハンドル。ターゲット変更の検出に使用。
    /// ターゲットが変更された場合、SOCP出力は未検証であり、
    /// 次のステップで検証されるまで収束を主張できない。
    pub prev_target_handles: Option<Vec<Vector2<f64>>>,
}

/// 効率化のための事前計算データ（Algorithm 1 "if first step" で計算）。
pub struct PrecomputedData {
    /// 全 z ∈ Z に対する f_i(z)、形状: (num_collocation, num_basis)
    pub phi: DMatrix<f64>,

    /// 各 z での ∂f_i/∂x、形状: (num_collocation, num_basis)
    pub grad_phi_x: DMatrix<f64>,

    /// 各 z での ∂f_i/∂y、形状: (num_collocation, num_basis)
    pub grad_phi_y: DMatrix<f64>,

    /// Biharmonic二次形式行列 (Eq. 31、数値積分)
    /// 形状: (2*n, 2*n)、n = num_basis
    pub biharmonic_matrix: Option<DMatrix<f64>>,
}

/// ソルバーからのエラー。
#[derive(Debug)]
pub enum SolverError {
    /// SOCPソルバーが実行可能解を見つけられなかった
    Infeasible(String),
    /// 問題構築における数値的問題
    NumericalError(String),
    /// ソルバーが予期しないステータスを返した
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

/// アルゴリズムからのエラー。
#[derive(Debug)]
pub enum AlgorithmError {
    Solver(SolverError),
    InvalidInput(String),
    /// Strategy 2 が設定された最大値 (`SolverConfig::max_refinement_resolution`)
    /// を超えるグリッド解像度を必要とする。
    /// 呼び出し側（例: UI）は必要な解像度で続行するかユーザーに確認すべき。
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

/// Algorithm 1 の入力パラメータ（歪み種類非依存）。
///
/// 歪み尺度（等長 vs 等角）に依存しないパラメータを含む。
/// 歪み固有の振る舞いは `DistortionPolicy` トレイトが担当する。
#[derive(Debug, Clone)]
pub struct MappingParams {
    /// 歪み上界 K (Eq. 4: D(z_j) ≤ K)
    pub k_bound: f64,

    /// 正則化重み λ (Eq. 1, 18)
    pub lambda_reg: f64,

    /// 正則化の種類と混合重み
    pub regularization: RegularizationType,
}

/// 各アルゴリズムステップから返される情報。
#[derive(Debug)]
pub struct StepInfo {
    /// 全コロケーション点での最大歪み（SOCPソルブ前に評価）
    pub max_distortion: f64,
    /// アクティブセット Z' の点数
    pub active_set_size: usize,
    /// 安定化セット Z'' の点数
    pub stable_set_size: usize,
    /// Algorithm 1 の収束: max_distortion ≤ K かつアクティブセット不変。
    ///
    /// `true` の場合、ソルブ前の歪みが既に上界内であり、
    /// 新たな制約点が不要だったことを意味する。つまり前回の
    /// SOCP解が全コロケーション点で歪み上界を満たしている。
    pub converged: bool,
}
