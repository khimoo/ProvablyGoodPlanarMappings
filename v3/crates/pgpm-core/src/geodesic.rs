//! Geodesic distance computation via Fast Marching Method (FMM).
//!
//! Paper Section "Shape aware bases" (p.76:9):
//! > "we tested shape aware variation of Gaussians, which is achieved by
//! > simply replacing the norm in their definition with the shortest
//! > distance function."
//!
//! The FMM solves the eikonal equation |∇T| = 1 on a 2D grid,
//! where cells outside the domain polygon are treated as obstacles.
//! This gives the shortest interior-path distance from a source point
//! to every grid cell.

use crate::types::DomainBounds;
use nalgebra::Vector2;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// A geodesic distance field computed by FMM from a single source.
pub struct GeodesicField {
    /// Distance values on the grid (row-major: idx = row * width + col).
    distances: Vec<f64>,
    /// Grid dimensions.
    width: usize,
    height: usize,
    /// Grid spacing.
    dx: f64,
    dy: f64,
    /// Domain bounds (for coordinate ↔ index conversion).
    x_min: f64,
    y_min: f64,
}

/// Cell state for FMM.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CellState {
    /// Not yet reached.
    Far,
    /// In the narrow band (tentative distance assigned).
    Narrow,
    /// Distance finalized.
    Frozen,
    /// Outside the domain (obstacle).
    Wall,
}

/// Priority queue entry (min-heap via Reverse ordering).
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
        // Reverse ordering for min-heap (BinaryHeap is max-heap by default)
        debug_assert!(
            !self.dist.is_nan() && !other.dist.is_nan(),
            "NaN distance in FMM priority queue (indices: {}, {})",
            self.idx, other.idx,
        );
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

impl GeodesicField {
    /// Compute the geodesic distance field from `source` on a grid.
    ///
    /// `domain_mask` should have length `width * height` (row-major),
    /// where `true` means the cell is inside the domain.
    /// The grid covers `bounds` with the given `width` × `height` cells.
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

        // Mark walls
        for i in 0..n {
            if !domain_mask[i] {
                states[i] = CellState::Wall;
            }
        }

        // Find the grid cell closest to source
        let src_col_f = (source.x - bounds.x_min) / dx;
        let src_row_f = (source.y - bounds.y_min) / dy;

        // Initialize seed cells: the 2×2 neighborhood around the source,
        // setting distance to the exact Euclidean distance from source.
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

        // FMM main loop
        while let Some(entry) = heap.pop() {
            let idx = entry.idx;
            if states[idx] == CellState::Frozen {
                continue;
            }
            states[idx] = CellState::Frozen;

            let row = idx / width;
            let col = idx % width;

            // Update 4-connected neighbors
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

    /// Get distance at a grid index.
    #[inline]
    pub fn distance_at_index(&self, idx: usize) -> f64 {
        self.distances[idx]
    }

    /// Bilinear interpolation of the distance field at an arbitrary point.
    ///
    /// If any of the 2×2 bilinear neighbors have Inf distance (Wall cells),
    /// they are excluded and only finite neighbors contribute. This prevents
    /// Inf from contaminating interpolation near the domain boundary.
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

        // Check if all four are finite → standard bilinear
        if d00.is_finite() && d10.is_finite() && d01.is_finite() && d11.is_finite() {
            let d0 = d00 * (1.0 - tc) + d10 * tc;
            let d1 = d01 * (1.0 - tc) + d11 * tc;
            return d0 * (1.0 - tr) + d1 * tr;
        }

        // Some neighbors are walls: weighted average of finite neighbors only
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

    /// Gradient of the distance field at a grid point (central differences).
    /// Returns (∂d/∂x, ∂d/∂y).
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

    /// Interpolated gradient at an arbitrary point.
    ///
    /// Skips Wall cells (Inf distance) when interpolating, analogous to
    /// `interpolate()`.
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

    /// Grid dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Grid spacing.
    pub fn spacing(&self) -> (f64, f64) {
        (self.dx, self.dy)
    }
}

/// Solve the 2D eikonal equation at cell (row, col) using upwind neighbors.
///
/// Standard FMM update: solve ((T - T_x)/dx)² + ((T - T_y)/dy)² = 1
/// where T_x, T_y are the minimum frozen x- and y-neighbor distances.
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
    // Get minimum frozen/narrow neighbor in x-direction
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

    // Get minimum frozen/narrow neighbor in y-direction
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

    // Solve: ((T - tx)/dx)² + ((T - ty)/dy)² = 1
    let a = 1.0 / (dx * dx) + 1.0 / (dy * dy);
    let b = -2.0 * (tx / (dx * dx) + ty / (dy * dy));
    let c = tx * tx / (dx * dx) + ty * ty / (dy * dy) - 1.0;

    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        // Fall back to 1D solution
        (tx + dx).min(ty + dy)
    } else {
        let t = (-b + discriminant.sqrt()) / (2.0 * a);
        // Ensure result is >= both neighbors
        if t >= tx && t >= ty {
            t
        } else {
            (tx + dx).min(ty + dy)
        }
    }
}

/// Build a domain mask on a grid from a polygon contour.
///
/// Returns a Vec<bool> of length width*height (row-major).
/// `true` if the grid cell at (col, row) is inside the polygon
/// **or within one grid cell of the polygon boundary**.
///
/// The standard ray-casting point-in-polygon test returns `false` for
/// points exactly on the boundary. Since the FMM marks non-interior
/// cells as walls (distance = Inf), this causes bilinear interpolation
/// to produce Inf near the boundary. Including boundary-adjacent cells
/// ensures the distance field is well-defined everywhere inside the
/// contour.
pub fn build_domain_mask(
    bounds: &DomainBounds,
    width: usize,
    height: usize,
    polygon: &[Vector2<f64>],
    holes: &[&[Vector2<f64>]],
) -> Vec<bool> {
    let dx = (bounds.x_max - bounds.x_min) / (width as f64 - 1.0);
    let dy = (bounds.y_max - bounds.y_min) / (height as f64 - 1.0);
    let margin = dx.max(dy) * 1.5; // include cells within 1.5 grid spacings of boundary

    let mut mask = vec![false; width * height];

    for row in 0..height {
        for col in 0..width {
            let x = bounds.x_min + col as f64 * dx;
            let y = bounds.y_min + row as f64 * dy;

            let in_outer = point_in_polygon_f64(x, y, polygon)
                || point_near_polygon_edge(x, y, polygon, margin);

            // Exclude points inside hole interiors (but keep those near hole edges
            // for smooth distance field transition)
            let in_hole = holes.iter().any(|hole| {
                point_in_polygon_f64(x, y, hole)
                    && !point_near_polygon_edge(x, y, hole, margin)
            });

            mask[row * width + col] = in_outer && !in_hole;
        }
    }

    mask
}

/// Check whether a point is within `margin` distance of any polygon edge.
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
        // Project point onto the line segment [a, b]
        let abx = bx - ax;
        let aby = by - ay;
        let len_sq = abx * abx + aby * aby;
        if len_sq < 1e-30 {
            // Degenerate edge
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

/// Ray-casting point-in-polygon test.
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
        // Simple square domain, no obstacles
        let bounds = DomainBounds {
            x_min: 0.0, x_max: 1.0,
            y_min: 0.0, y_max: 1.0,
        };
        let w = 21;
        let h = 21;
        let mask = vec![true; w * h];
        let source = Vector2::new(0.5, 0.5);

        let field = GeodesicField::compute(source, &bounds, w, h, &mask);

        // Distance at source should be ~0
        let center_idx = 10 * w + 10;
        assert!(field.distance_at_index(center_idx) < 0.1);

        // Distance at corner (0,0) should be ~sqrt(0.5) ≈ 0.707
        let corner_dist = field.distance_at_index(0);
        assert!((corner_dist - 0.5_f64.sqrt()).abs() < 0.1,
            "Corner distance: {}, expected ~{}", corner_dist, 0.5_f64.sqrt());
    }

    #[test]
    fn test_fmm_with_wall() {
        // Domain with a wall that forces a detour
        let bounds = DomainBounds {
            x_min: 0.0, x_max: 4.0,
            y_min: 0.0, y_max: 4.0,
        };
        let w = 41;
        let h = 41;
        let mut mask = vec![true; w * h];

        // Create a wall at x=2, from y=0 to y=3 (leaving a gap at y=3..4)
        let wall_col = 20; // x = 2.0
        for row in 0..31 { // y = 0.0 to 3.0
            mask[row * w + wall_col] = false;
        }

        let source = Vector2::new(1.0, 1.0);
        let field = GeodesicField::compute(source, &bounds, w, h, &mask);

        // Point at (3, 1) is on the other side of the wall.
        // Euclidean distance = 2.0, but geodesic must go around the wall.
        let target_col = 30; // x = 3.0
        let target_row = 10; // y = 1.0
        let geodesic_dist = field.distance_at_index(target_row * w + target_col);

        // Geodesic distance should be significantly greater than Euclidean
        assert!(geodesic_dist > 2.5,
            "Geodesic distance {} should be > 2.5 (Euclidean = 2.0)", geodesic_dist);
    }
}
