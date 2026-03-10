//! Domain abstraction for the algorithm.
//!
//! Paper Section 3-4: the algorithm controls distortion and injectivity
//! over a domain Ω ⊂ R². This module provides the `Domain` trait that
//! abstracts the "x ∈ Ω" test, and a polygon-based implementation.

use nalgebra::Vector2;

/// Abstract domain Ω.
///
/// Paper Section 4: "consider all the points from a surrounding uniform
/// grid that fall inside the domain"
///
/// The algorithm only needs to test whether a point belongs to Ω.
/// How the domain is defined (polygon, alpha channel, SDF, etc.) is
/// the caller's concern.
pub trait Domain: Send + Sync {
    /// Test if point `pt` is inside the domain Ω.
    fn contains(&self, pt: &Vector2<f64>) -> bool;
}

/// Domain defined by an outer polygon boundary with optional holes.
///
/// A point is inside the domain if it is inside the outer contour
/// and outside all hole contours.
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

/// Ray-casting point-in-polygon test.
///
/// Paper Section 4: "consider all the points from a surrounding uniform
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
        // Outer: 10×10 square
        let outer = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(10.0, 0.0),
            Vector2::new(10.0, 10.0),
            Vector2::new(0.0, 10.0),
        ];
        // Hole: 3×3 square centered at (5, 5)
        let hole = vec![
            Vector2::new(3.5, 3.5),
            Vector2::new(6.5, 3.5),
            Vector2::new(6.5, 6.5),
            Vector2::new(3.5, 6.5),
        ];
        let domain = PolygonDomain::new(outer, vec![hole]);

        // Inside outer, outside hole → inside domain
        assert!(domain.contains(&Vector2::new(1.0, 1.0)));
        // Inside outer, inside hole → outside domain
        assert!(!domain.contains(&Vector2::new(5.0, 5.0)));
        // Outside outer → outside domain
        assert!(!domain.contains(&Vector2::new(15.0, 5.0)));
    }
}
