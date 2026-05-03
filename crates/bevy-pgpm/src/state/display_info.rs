//! 表示情報リソース。各アルゴリズムステップで更新。

use bevy::prelude::*;

/// UI 表示情報。各アルゴリズムステップで更新。
///
/// pgpm-core からのステップ毎の結果のみを含む。アルゴリズムパラメータ
/// （K、lambda、正則化）は [`super::AlgoParams`] から直接読み取る。
#[derive(Resource, Default)]
pub struct DeformationInfo {
    pub max_distortion: f64,
    pub active_set_size: usize,
    pub stable_set_size: usize,
    pub step_count: usize,
    /// pgpm-core からの Algorithm 1 収束フラグ。
    pub converged: bool,
    /// Strategy 2 結果のステータスメッセージ（None = 未実行）
    pub strategy2_status: Option<String>,
}
