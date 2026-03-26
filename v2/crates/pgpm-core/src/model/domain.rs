//! アルゴリズム用のドメイン抽象化。
//!
//! 論文 Section 3-4: アルゴリズムはドメイン Ω ⊂ R² 上で
//! 歪みと単射性を制御する。このモジュールは "x ∈ Ω" の判定を
//! 抽象化する `Domain` トレイトと、ポリゴンベースの実装を提供する。

use nalgebra::Vector2;

/// 抽象ドメイン Ω。
///
/// 論文 Section 4: "consider all the points from a surrounding uniform
/// grid that fall inside the domain"
///
/// アルゴリズムが必要とするのは、点が Ω に属するかどうかの判定のみ。
/// ドメインの定義方法（ポリゴン、アルファチャンネル、SDF等）は
/// 呼び出し側の責任。
pub trait Domain: Send + Sync {
    /// 点 `pt` がドメイン Ω の内部にあるかを判定する。
    fn contains(&self, pt: &Vector2<f64>) -> bool;
}

/// 外側ポリゴン境界とオプションの穴で定義されるドメイン。
///
/// 点がドメイン内にあるとは、外側輪郭の内部かつ
/// 全ての穴輪郭の外部にあること。
pub struct PolygonDomain {
    outer: Vec<Vector2<f64>>,
    holes: Vec<Vec<Vector2<f64>>>,
}

impl PolygonDomain {
    pub fn new(outer: Vec<Vector2<f64>>, holes: Vec<Vec<Vector2<f64>>>) -> Self {
        Self { outer, holes }
    }
}

impl Domain for PolygonDomain {
    fn contains(&self, pt: &Vector2<f64>) -> bool {
        point_in_polygon(pt, &self.outer)
            && !self.holes.iter().any(|hole| point_in_polygon(pt, hole))
    }
}

/// レイキャスティングによる点のポリゴン内包判定。
///
/// 論文 Section 4: "consider all the points from a surrounding uniform
/// grid that fall inside the domain"
pub(crate) fn point_in_polygon(pt: &Vector2<f64>, polygon: &[Vector2<f64>]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let (x, y) = (pt.x, pt.y);
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
    fn test_point_in_polygon_square() {
        let square = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(0.0, 1.0),
        ];
        assert!(point_in_polygon(&Vector2::new(0.5, 0.5), &square));
        assert!(!point_in_polygon(&Vector2::new(1.5, 0.5), &square));
        assert!(!point_in_polygon(&Vector2::new(-0.5, 0.5), &square));
    }

    #[test]
    fn test_point_in_polygon_triangle() {
        let tri = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(0.5, 1.0),
        ];
        assert!(point_in_polygon(&Vector2::new(0.5, 0.3), &tri));
        assert!(!point_in_polygon(&Vector2::new(0.1, 0.9), &tri));
    }

    #[test]
    fn test_polygon_domain_with_hole() {
        // 外側: 10×10の正方形
        let outer = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(10.0, 0.0),
            Vector2::new(10.0, 10.0),
            Vector2::new(0.0, 10.0),
        ];
        // 穴: (5, 5)を中心とした3×3の正方形
        let hole = vec![
            Vector2::new(3.5, 3.5),
            Vector2::new(6.5, 3.5),
            Vector2::new(6.5, 6.5),
            Vector2::new(3.5, 6.5),
        ];
        let domain = PolygonDomain::new(outer, vec![hole]);

        // 外側の内部、穴の外部 → ドメイン内
        assert!(domain.contains(&Vector2::new(1.0, 1.0)));
        // 外側の内部、穴の内部 → ドメイン外
        assert!(!domain.contains(&Vector2::new(5.0, 5.0)));
        // 外側の外部 → ドメイン外
        assert!(!domain.contains(&Vector2::new(15.0, 5.0)));
    }
}
