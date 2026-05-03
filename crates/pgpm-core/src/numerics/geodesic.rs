//! Fast Marching Method (FMM) による測地距離計算。
//!
//! 論文 Section "Shape aware bases" (p.76:9):
//! > "we tested shape aware variation of Gaussians, which is achieved by
//! > simply replacing the norm in their definition with the shortest
//! > distance function."
//!
//! FMM は2Dグリッド上でアイコナル方程式 |∇T| = 1 を解く。
//! ドメインポリゴン外のセルは障害物として扱う。
//! これにより、ソース点から全グリッドセルへのドメイン内部最短経路距離が得られる。

use crate::model::types::DomainBounds;
use nalgebra::Vector2;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// 単一ソースからFMMで計算された測地距離場。
pub struct GeodesicField {
    /// グリッド上の距離値（行優先: idx = row * width + col）。
    distances: Vec<f64>,
    /// グリッドの次元。
    width: usize,
    height: usize,
    /// グリッド間隔。
    dx: f64,
    dy: f64,
    /// ドメイン境界（座標 ↔ インデックス変換用）。
    x_min: f64,
    y_min: f64,
}

/// FMM用のセル状態。
#[derive(Clone, Copy, PartialEq, Eq)]
enum CellState {
    /// 未到達。
    Far,
    /// ナローバンド内（暫定距離が割り当て済み）。
    Narrow,
    /// 距離が確定済み。
    Frozen,
    /// ドメイン外（障害物）。
    Wall,
}

/// 優先度キューのエントリ（Reverseによる最小ヒープ）。
#[derive(Clone, Copy)]
struct FmmEntry {
    dist: f64,
    idx: usize,
}

impl PartialEq for FmmEntry {
    fn eq(&self, other: &Self) -> bool { self.idx == other.idx }
}
impl Eq for FmmEntry {}

impl PartialOrd for FmmEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for FmmEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // 最小ヒープのための逆順（BinaryHeap はデフォルトで最大ヒープ）
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

impl GeodesicField {
    /// `source` からの測地距離場をグリッド上で計算する。
    ///
    /// `domain_mask` は長さ `width * height`（行優先）で、
    /// `true` はセルがドメイン内部であることを意味する。
    /// グリッドは指定された `width` × `height` のセルで `bounds` を覆う。
    pub fn compute(
        source: Vector2<f64>,
        bounds: &DomainBounds,
        width: usize,
        height: usize,
        domain_mask: &[bool],
    ) -> Self {
        assert_eq!(domain_mask.len(), width * height);

        let dx = (bounds.x_max - bounds.x_min) / (width as f64 - 1.0);
        let dy = (bounds.y_max - bounds.y_min) / (height as f64 - 1.0);
        let n = width * height;

        let mut distances = vec![f64::INFINITY; n];
        let mut states = vec![CellState::Far; n];

        // 壁をマーク
        for i in 0..n {
            if !domain_mask[i] {
                states[i] = CellState::Wall;
            }
        }

        // ソースに最も近いグリッドセルを見つける
        let src_col_f = (source.x - bounds.x_min) / dx;
        let src_row_f = (source.y - bounds.y_min) / dy;

        // シードセルの初期化: ソース周りの2×2近傍、
        // ソースからの正確なユークリッド距離を設定
        let mut heap = BinaryHeap::new();

        let col0 = (src_col_f.floor() as isize).max(0) as usize;
        let row0 = (src_row_f.floor() as isize).max(0) as usize;

        for dr in 0..=1usize {
            for dc in 0..=1usize {
                let r = (row0 + dr).min(height - 1);
                let c = (col0 + dc).min(width - 1);
                let idx = r * width + c;
                if states[idx] == CellState::Wall {
                    continue;
                }
                let cell_x = bounds.x_min + c as f64 * dx;
                let cell_y = bounds.y_min + r as f64 * dy;
                let d = ((cell_x - source.x).powi(2) + (cell_y - source.y).powi(2)).sqrt();
                if d < distances[idx] {
                    distances[idx] = d;
                    states[idx] = CellState::Narrow;
                    heap.push(FmmEntry { dist: d, idx });
                }
            }
        }

        // FMMメインループ
        while let Some(entry) = heap.pop() {
            let idx = entry.idx;
            if states[idx] == CellState::Frozen {
                continue;
            }
            states[idx] = CellState::Frozen;

            let row = idx / width;
            let col = idx % width;

            // 4連結近傍を更新
            let neighbors: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            for &(dr, dc) in &neighbors {
                let nr = row as isize + dr;
                let nc = col as isize + dc;
                if nr < 0 || nr >= height as isize || nc < 0 || nc >= width as isize {
                    continue;
                }
                let nidx = nr as usize * width + nc as usize;
                if states[nidx] == CellState::Frozen || states[nidx] == CellState::Wall {
                    continue;
                }

                let new_dist = solve_eikonal_2d(
                    &distances, &states, nr as usize, nc as usize,
                    width, height, dx, dy,
                );

                if new_dist < distances[nidx] {
                    distances[nidx] = new_dist;
                    states[nidx] = CellState::Narrow;
                    heap.push(FmmEntry { dist: new_dist, idx: nidx });
                }
            }
        }

        Self {
            distances,
            width,
            height,
            dx,
            dy,
            x_min: bounds.x_min,
            y_min: bounds.y_min,
        }
    }

    /// グリッドインデックスでの距離を取得する。
    #[inline]
    pub fn distance_at_index(&self, idx: usize) -> f64 {
        self.distances[idx]
    }

    /// 任意の点での距離場の双線形補間。
    ///
    /// 2×2双線形近傍のいずれかがInf距離（壁セル）の場合、
    /// 有限な近傍のみで寄与する。ドメイン境界付近で
    /// Infが補間を汚染するのを防ぐ。
    pub fn interpolate(&self, x: Vector2<f64>) -> f64 {
        let cf = (x.x - self.x_min) / self.dx;
        let rf = (x.y - self.y_min) / self.dy;

        let c0 = (cf.floor() as isize).max(0).min(self.width as isize - 2) as usize;
        let r0 = (rf.floor() as isize).max(0).min(self.height as isize - 2) as usize;
        let c1 = c0 + 1;
        let r1 = r0 + 1;

        let tc = (cf - c0 as f64).clamp(0.0, 1.0);
        let tr = (rf - r0 as f64).clamp(0.0, 1.0);

        let d00 = self.distances[r0 * self.width + c0];
        let d10 = self.distances[r0 * self.width + c1];
        let d01 = self.distances[r1 * self.width + c0];
        let d11 = self.distances[r1 * self.width + c1];

        // 4つ全てが有限 → 標準的な双線形補間
        if d00.is_finite() && d10.is_finite() && d01.is_finite() && d11.is_finite() {
            let d0 = d00 * (1.0 - tc) + d10 * tc;
            let d1 = d01 * (1.0 - tc) + d11 * tc;
            return d0 * (1.0 - tr) + d1 * tr;
        }

        // 一部の近傍が壁: 有限な近傍のみの加重平均
        let corners = [
            (d00, (1.0 - tc) * (1.0 - tr)),
            (d10, tc * (1.0 - tr)),
            (d01, (1.0 - tc) * tr),
            (d11, tc * tr),
        ];
        let mut sum = 0.0;
        let mut w_sum = 0.0;
        for &(d, w) in &corners {
            if d.is_finite() {
                sum += d * w;
                w_sum += w;
            }
        }
        if w_sum > 0.0 {
            sum / w_sum
        } else {
            f64::INFINITY
        }
    }

    /// グリッド点での距離場の勾配（中心差分）。
    /// (∂d/∂x, ∂d/∂y) を返す。
    pub fn gradient_at_index(&self, idx: usize) -> Vector2<f64> {
        let row = idx / self.width;
        let col = idx % self.width;

        let ddx = if col == 0 {
            (self.distances[idx + 1] - self.distances[idx]) / self.dx
        } else if col == self.width - 1 {
            (self.distances[idx] - self.distances[idx - 1]) / self.dx
        } else {
            (self.distances[idx + 1] - self.distances[idx - 1]) / (2.0 * self.dx)
        };

        let ddy = if row == 0 {
            (self.distances[idx + self.width] - self.distances[idx]) / self.dy
        } else if row == self.height - 1 {
            (self.distances[idx] - self.distances[idx - self.width]) / self.dy
        } else {
            (self.distances[idx + self.width] - self.distances[idx - self.width]) / (2.0 * self.dy)
        };

        Vector2::new(ddx, ddy)
    }

    /// 任意の点での補間された勾配。
    ///
    /// `interpolate()` と同様に壁セル（Inf距離）をスキップする。
    pub fn interpolate_gradient(&self, x: Vector2<f64>) -> Vector2<f64> {
        let cf = (x.x - self.x_min) / self.dx;
        let rf = (x.y - self.y_min) / self.dy;

        let c0 = (cf.floor() as isize).max(0).min(self.width as isize - 2) as usize;
        let r0 = (rf.floor() as isize).max(0).min(self.height as isize - 2) as usize;
        let c1 = c0 + 1;
        let r1 = r0 + 1;

        let tc = (cf - c0 as f64).clamp(0.0, 1.0);
        let tr = (rf - r0 as f64).clamp(0.0, 1.0);

        let indices = [
            (r0 * self.width + c0, (1.0 - tc) * (1.0 - tr)),
            (r0 * self.width + c1, tc * (1.0 - tr)),
            (r1 * self.width + c0, (1.0 - tc) * tr),
            (r1 * self.width + c1, tc * tr),
        ];

        let mut gx = 0.0;
        let mut gy = 0.0;
        let mut w_sum = 0.0;

        for &(idx, w) in &indices {
            if self.distances[idx].is_finite() {
                let g = self.gradient_at_index(idx);
                if g.x.is_finite() && g.y.is_finite() {
                    gx += g.x * w;
                    gy += g.y * w;
                    w_sum += w;
                }
            }
        }

        if w_sum > 0.0 {
            Vector2::new(gx / w_sum, gy / w_sum)
        } else {
            Vector2::new(0.0, 0.0)
        }
    }

    /// グリッドの次元。
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// グリッドの間隔。
    pub fn spacing(&self) -> (f64, f64) {
        (self.dx, self.dy)
    }
}

/// セル (row, col) での2Dアイコナル方程式を風上近傍で解く。
///
/// 標準FMM更新: ((T - T_x)/dx)² + ((T - T_y)/dy)² = 1 を解く。
/// T_x, T_y は最小の確定済みx/y方向近傍距離。
fn solve_eikonal_2d(
    distances: &[f64],
    states: &[CellState],
    row: usize,
    col: usize,
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
) -> f64 {
    // x方向の最小確定/ナローバンド近傍を取得
    let tx = {
        let left = if col > 0 {
            let i = row * width + col - 1;
            if states[i] == CellState::Frozen { distances[i] } else { f64::INFINITY }
        } else { f64::INFINITY };
        let right = if col + 1 < width {
            let i = row * width + col + 1;
            if states[i] == CellState::Frozen { distances[i] } else { f64::INFINITY }
        } else { f64::INFINITY };
        left.min(right)
    };

    // y方向の最小確定/ナローバンド近傍を取得
    let ty = {
        let up = if row > 0 {
            let i = (row - 1) * width + col;
            if states[i] == CellState::Frozen { distances[i] } else { f64::INFINITY }
        } else { f64::INFINITY };
        let down = if row + 1 < height {
            let i = (row + 1) * width + col;
            if states[i] == CellState::Frozen { distances[i] } else { f64::INFINITY }
        } else { f64::INFINITY };
        up.min(down)
    };

    if tx.is_infinite() && ty.is_infinite() {
        return f64::INFINITY;
    }

    if tx.is_infinite() {
        return ty + dy;
    }
    if ty.is_infinite() {
        return tx + dx;
    }

    // 解く: ((T - tx)/dx)² + ((T - ty)/dy)² = 1
    let a = 1.0 / (dx * dx) + 1.0 / (dy * dy);
    let b = -2.0 * (tx / (dx * dx) + ty / (dy * dy));
    let c = tx * tx / (dx * dx) + ty * ty / (dy * dy) - 1.0;

    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        // 1D解にフォールバック
        (tx + dx).min(ty + dy)
    } else {
        let t = (-b + discriminant.sqrt()) / (2.0 * a);
        // 結果が両方の近傍以上であることを確認
        if t >= tx && t >= ty {
            t
        } else {
            (tx + dx).min(ty + dy)
        }
    }
}

/// ポリゴン輪郭からグリッド上のドメインマスクを構築する。
///
/// 長さ width*height の Vec<bool>（行優先）を返す。
/// (col, row) のグリッドセルがポリゴンの内部
/// **またはポリゴン境界から1グリッドセル以内**にある場合に `true`。
///
/// 標準的なレイキャスティング内包判定は境界上の点に対して `false` を返す。
/// FMMが非内部セルを壁（距離 = Inf）としてマークするため、
/// 境界付近の双線形補間がInfになってしまう。境界隣接セルを含めることで、
/// 輪郭内部の全箇所で距離場が適切に定義されることを保証する。
pub fn build_domain_mask(
    bounds: &DomainBounds,
    width: usize,
    height: usize,
    polygon: &[Vector2<f64>],
) -> Vec<bool> {
    let dx = (bounds.x_max - bounds.x_min) / (width as f64 - 1.0);
    let dy = (bounds.y_max - bounds.y_min) / (height as f64 - 1.0);
    let margin = dx.max(dy) * 1.5; // 境界から1.5グリッド間隔以内のセルを含める

    let mut mask = vec![false; width * height];

    for row in 0..height {
        for col in 0..width {
            let x = bounds.x_min + col as f64 * dx;
            let y = bounds.y_min + row as f64 * dy;
            mask[row * width + col] =
                point_in_polygon_f64(x, y, polygon)
                || point_near_polygon_edge(x, y, polygon, margin);
        }
    }

    mask
}

/// 点がポリゴン辺から `margin` 距離以内にあるかを判定する。
fn point_near_polygon_edge(x: f64, y: f64, polygon: &[Vector2<f64>], margin: f64) -> bool {
    let n = polygon.len();
    if n < 2 {
        return false;
    }
    let margin_sq = margin * margin;
    for i in 0..n {
        let j = (i + 1) % n;
        let (ax, ay) = (polygon[i].x, polygon[i].y);
        let (bx, by) = (polygon[j].x, polygon[j].y);
        // 点を線分 [a, b] に射影
        let abx = bx - ax;
        let aby = by - ay;
        let len_sq = abx * abx + aby * aby;
        if len_sq < 1e-30 {
            // 退化した辺
            let dx = x - ax;
            let dy = y - ay;
            if dx * dx + dy * dy <= margin_sq {
                return true;
            }
            continue;
        }
        let t = ((x - ax) * abx + (y - ay) * aby) / len_sq;
        let t_clamped = t.clamp(0.0, 1.0);
        let px = ax + t_clamped * abx;
        let py = ay + t_clamped * aby;
        let dx = x - px;
        let dy = y - py;
        if dx * dx + dy * dy <= margin_sq {
            return true;
        }
    }
    false
}

/// レイキャスティングによる点のポリゴン内包判定。
fn point_in_polygon_f64(x: f64, y: f64, polygon: &[Vector2<f64>]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (polygon[i].x, polygon[i].y);
        let (xj, yj) = (polygon[j].x, polygon[j].y);
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmm_open_domain() {
        // 単純な正方形ドメイン、障害物なし
        let bounds = DomainBounds {
            x_min: 0.0, x_max: 1.0,
            y_min: 0.0, y_max: 1.0,
        };
        let w = 21;
        let h = 21;
        let mask = vec![true; w * h];
        let source = Vector2::new(0.5, 0.5);

        let field = GeodesicField::compute(source, &bounds, w, h, &mask);

        // ソースでの距離は ~0 であるべき
        let center_idx = 10 * w + 10;
        assert!(field.distance_at_index(center_idx) < 0.1);

        // 角 (0,0) での距離は ~sqrt(0.5) ≈ 0.707 であるべき
        let corner_dist = field.distance_at_index(0);
        assert!((corner_dist - 0.5_f64.sqrt()).abs() < 0.1,
            "Corner distance: {}, expected ~{}", corner_dist, 0.5_f64.sqrt());
    }

    #[test]
    fn test_fmm_with_wall() {
        // 迂回を強いる壁のあるドメイン
        let bounds = DomainBounds {
            x_min: 0.0, x_max: 4.0,
            y_min: 0.0, y_max: 4.0,
        };
        let w = 41;
        let h = 41;
        let mut mask = vec![true; w * h];

        // x=2 に y=0 から y=3 までの壁を作成（y=3..4 にギャップを残す）
        let wall_col = 20; // x = 2.0
        for row in 0..31 { // y = 0.0 から 3.0
            mask[row * w + wall_col] = false;
        }

        let source = Vector2::new(1.0, 1.0);
        let field = GeodesicField::compute(source, &bounds, w, h, &mask);

        // (3, 1) の点は壁の反対側にある。
        // ユークリッド距離 = 2.0 だが、測地距離は壁を迂回する必要がある。
        let target_col = 30; // x = 3.0
        let target_row = 10; // y = 1.0
        let geodesic_dist = field.distance_at_index(target_row * w + target_col);

        // 測地距離はユークリッド距離より大幅に大きいはず
        assert!(geodesic_dist > 2.5,
            "Geodesic distance {} should be > 2.5 (Euclidean = 2.0)", geodesic_dist);
    }
}
