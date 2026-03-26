//! アクティブ集合管理。
//!
//! 論文参照:
//! - Algorithm 1: 初期化とアクティブ集合更新
//! - Section 5 "Activation of constraints":
//!   「歪みの局所最大を見つける」
//!   「D(z) > K_high の点のみを追加」
//!   「D(z) < K_low の点を削除」

use crate::model::types::AlgorithmState;
use nalgebra::Vector2;

/// Algorithm 1: アクティブ集合 Z' を更新。
///
/// 論文に厳密に従う:
/// 1. グリッド上の歪みの局所最大 Z_max を見つける
/// 2. D(z) > K_high となる z ∈ Z_max を Z' に挿入
/// 3. D(z) < K_low となる z ∈ Z' を Z' から削除
///
/// **重要**: fold-over 予防、σ チェック、サイズ制限なし。
/// 論文は局所最大フィルタで十分としている。
pub fn update_active_set(
    state: &mut AlgorithmState,
    distortions: &[f64],
) {
    // ステップ1: グリッド上の局所最大を見つける
    let local_maxima = find_local_maxima(distortions, state.grid_width, state.grid_height);

    // ステップ2: K_high を超える局所最大を追加
    // 「D(z) > K_high となる各 z ∈ Z_max を Z' に挿入」
    // 論文 Section 4: ドメイン内の点のみが制約される。
    for &idx in &local_maxima {
        if distortions[idx] > state.k_high && state.domain_mask[idx] {
            if !state.active_set.contains(&idx) {
                state.active_set.push(idx);
            }
        }
    }

    // ステップ3: K_low 未満の点を削除
    // 「D(z) < K_low となる各 z ∈ Z' を Z' から削除」
    state.active_set.retain(|&idx| distortions[idx] >= state.k_low);
}

/// 矩形グリッド上の局所最大を検出（8近傍比較）。
///
/// Section 5: 「歪みの局所最大を見つける」
/// 「コロケーション点は密な矩形グリッド上でサンプリングされる」
///
/// 点は、その歪みが全近傍（グリッド上で最大8つ）より
/// 厳密に大きい場合に局所最大となる。
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

            // 全8近傍をチェック
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

/// Algorithm 1: 「最遠点サンプリングで集合 Z'' を初期化」
///
/// コロケーション点から **ドメイン内に存在する** k 個の
/// 等間隔点を選択する最遠点サンプリング（FPS）。
///
/// Section 5: 「等間隔に配置されたコロケーション点の
/// 小さな部分集合を常にアクティブに保つことができる」
pub fn initialize_stable_set(
    collocation_points: &[Vector2<f64>],
    k: usize,
    domain_mask: &[bool],
) -> Vec<usize> {
    let n = collocation_points.len();
    // 候補インデックス: ドメイン内部点のみ
    let candidates: Vec<usize> = (0..n).filter(|&i| domain_mask[i]).collect();
    if k == 0 || candidates.is_empty() {
        return Vec::new();
    }
    if k >= candidates.len() {
        return candidates;
    }

    let mut selected = Vec::with_capacity(k);
    let mut min_dists = vec![f64::INFINITY; n];

    // 最初の候補から開始
    selected.push(candidates[0]);

    for _ in 1..k {
        // 最後に選択された点に基づいて最小距離を更新
        let last = *selected.last().unwrap();
        let last_pt = collocation_points[last];
        for &i in &candidates {
            let d = (collocation_points[i] - last_pt).norm_squared();
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
        // 再選択を避けるため、既選択点の距離を0に設定
        for &s in &selected {
            min_dists[s] = 0.0;
        }

        // 最小距離が最大となる候補を選択
        let best = candidates.iter()
            .copied()
            .filter(|i| !selected.contains(i))
            .max_by(|&a, &b| min_dists[a].total_cmp(&min_dists[b]))
            .expect("FPS: no remaining candidates despite k < candidates.len()");

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
        // 中央に単一のピークを持つ3x3グリッド
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
        // コーナーのピーク
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
        // プラトー: 厳密な局所最大なし
        #[rustfmt::skip]
        let distortions = vec![
            1.0, 1.0, 1.0,
            1.0, 5.0, 5.0,
            1.0, 1.0, 1.0,
        ];
        let maxima = find_local_maxima(&distortions, 3, 3);
        // (1,1) も (1,2) も等しいため厳密な局所最大ではない
        assert!(maxima.is_empty());
    }

    #[test]
    fn test_active_set_update() {
        let mut state = crate::model::types::AlgorithmState {
            coefficients: nalgebra::DMatrix::zeros(2, 1),
            collocation_points: Vec::new(),
            active_set: Vec::new(),
            stable_set: Vec::new(),
            frames: Vec::new(),
            k_high: 2.8, // 0.1 + 0.9*3
            k_low: 2.0,   // 0.5 + 0.5*3
            domain_mask: vec![true; 9],
            precomputed: None,
            grid_width: 3,
            grid_height: 3,
            prev_target_handles: None,
        };

        // 3x3グリッド、中央に高い歪み
        #[rustfmt::skip]
        let distortions = vec![
            1.0, 1.0, 1.0,
            1.0, 3.5, 1.0,
            1.0, 1.0, 1.0,
        ];

        update_active_set(&mut state, &distortions);
        // 中央 (idx=4) は局所最大で D=3.5 > K_high=2.8
        assert!(state.active_set.contains(&4));
    }

    #[test]
    fn test_active_set_removal() {
        let mut state = crate::model::types::AlgorithmState {
            coefficients: nalgebra::DMatrix::zeros(2, 1),
            collocation_points: Vec::new(),
            active_set: vec![4],
            stable_set: Vec::new(),
            frames: Vec::new(),
            k_high: 2.8,
            k_low: 2.0,
            domain_mask: vec![true; 9],
            precomputed: None,
            grid_width: 3,
            grid_height: 3,
            prev_target_handles: None,
        };

        // 4の歪みがK_low以下に低下
        #[rustfmt::skip]
        let distortions = vec![
            1.0, 1.0, 1.0,
            1.0, 1.5, 1.0,
            1.0, 1.0, 1.0,
        ];

        update_active_set(&mut state, &distortions);
        // 点4は D=1.5 < K_low=2.0 なので削除されるべき
        assert!(!state.active_set.contains(&4));
    }

    #[test]
    fn test_fps_basic() {
        // 単位正方形の4隅
        let points = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(0.0, 1.0),
            Vector2::new(1.0, 1.0),
        ];
        let mask = vec![true; 4];

        let selected = initialize_stable_set(&points, 4, &mask);
        assert_eq!(selected.len(), 4);
        // 全4点が選択されるべき
        for i in 0..4 {
            assert!(selected.contains(&i));
        }
    }

    #[test]
    fn test_fps_selects_spread_points() {
        // 点の列、FPSは両端と中央を選ぶべき
        let points: Vec<Vector2<f64>> = (0..11)
            .map(|i| Vector2::new(i as f64, 0.0))
            .collect();
        let mask = vec![true; 11];

        let selected = initialize_stable_set(&points, 3, &mask);
        assert_eq!(selected.len(), 3);
        // 最初の点(0)と最遠(10)、次に中央(5)を含むべき
        assert!(selected.contains(&0));
        assert!(selected.contains(&10));
    }

    #[test]
    fn test_active_set_skips_exterior_points() {
        // 3x3グリッド: 中央はドメイン外 (mask = false)
        let mut state = crate::model::types::AlgorithmState {
            coefficients: nalgebra::DMatrix::zeros(2, 1),
            collocation_points: Vec::new(),
            active_set: Vec::new(),
            stable_set: Vec::new(),
            frames: Vec::new(),
            k_high: 2.8,
            k_low: 2.0,
            domain_mask: vec![true, true, true, true, false, true, true, true, true],
            precomputed: None,
            grid_width: 3,
            grid_height: 3,
            prev_target_handles: None,
        };

        // 中央は高い歪みを持つがドメイン外
        #[rustfmt::skip]
        let distortions = vec![
            1.0, 1.0, 1.0,
            1.0, 5.0, 1.0,
            1.0, 1.0, 1.0,
        ];

        update_active_set(&mut state, &distortions);
        // 中央 (idx=4) は局所最大で D=5.0 > K_high だが domain_mask[4]=false
        assert!(!state.active_set.contains(&4),
            "Exterior point should not be added to active set");
    }

    #[test]
    fn test_fps_skips_exterior_points() {
        // 5点だが、点2はドメイン外
        let points = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(2.0, 0.0), // 外側
            Vector2::new(3.0, 0.0),
            Vector2::new(4.0, 0.0),
        ];
        let mask = vec![true, true, false, true, true];

        let selected = initialize_stable_set(&points, 3, &mask);
        assert_eq!(selected.len(), 3);
        // 点2は選択されないべき
        assert!(!selected.contains(&2),
            "Exterior point should not be in stable set");
    }
}
