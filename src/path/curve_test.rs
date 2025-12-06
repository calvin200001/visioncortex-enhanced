use crate::{PathF64, PointF64, Spline};
use super::smooth::SubdivideSmooth;
use std::f64::consts::PI;

fn create_circle_path(radius: f64, steps: usize) -> PathF64 {
    let mut points = Vec::new();
    for i in 0..steps {
        let angle = 2.0 * PI * (i as f64) / (steps as f64);
        points.push(PointF64 {
            x: radius * angle.cos(),
            y: radius * angle.sin(),
        });
    }
    // Close the path
    points.push(points[0]);
    PathF64::from_points(points)
}

#[test]
fn test_fit_points_with_bezier_truncation() {
    // Create points for a full circle
    let path = create_circle_path(100.0, 32);
    let points = &path.path[0..path.path.len()-1]; // Remove closing point for fitting check if needed, 
                                                    // but fit_points_with_bezier expects a slice of points.
    
    // Direct call to fit_points_with_bezier
    // This function returns Vec<[PointF64; 4]>.
    let bezier = SubdivideSmooth::fit_points_with_bezier(points);
    
    // Check how many curves were returned
    println!("Bezier curves returned: {}", bezier.len());
    println!("Bezier control points: {:?}", bezier);
}

#[test]
fn test_spline_from_path_high_threshold() {
    let path = create_circle_path(100.0, 32);
    
    // Case A: High threshold (45.0 radians), effectively disabling splitting
    let splice_threshold_high = 45.0; 
    let spline_high = Spline::from_path_f64(&path, splice_threshold_high);
    
    println!("High threshold (45.0) produced {} curves", spline_high.num_curves());
    
    // Case B: Reasonable threshold (45 degrees = ~0.785 radians)
    let splice_threshold_reasonable = 45.0f64.to_radians();
    let spline_reasonable = Spline::from_path_f64(&path, splice_threshold_reasonable);

    println!("Reasonable threshold (0.785) produced {} curves", spline_reasonable.num_curves());

    // Expectation: 
    // High threshold -> 1 curve (Bad quality)
    // Reasonable threshold -> 4+ curves (Good quality)
}
