

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    
    pub fn distance_to(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

#[derive(Debug)]
pub struct CurvatureProfile {
    /// Curvature value at each point (higher = sharper turn)
    pub curvatures: Vec<f64>,
    /// Whether each point is a "feature point" (high artistic importance)
    pub feature_points: Vec<bool>,
    /// Suggested segment lengths for smoothing
    pub adaptive_lengths: Vec<f64>,
}

pub struct CurvatureAnalyzer {
    /// Base segment length (from VTracer's config)
    base_segment_length: f64,
    /// Window size for local curvature calculation
    window_size: usize,
    /// Threshold for detecting high-curvature features
    feature_threshold: f64,
}

impl CurvatureAnalyzer {
    pub fn new(base_segment_length: f64, window_size: usize, feature_threshold: f64) -> Self {
        Self {
            base_segment_length,
            window_size,
            feature_threshold,
        }
    }
    
    /// Calculate signed curvature at a point using discrete approximation
    /// This respects VTracer's discrete geometry approach
    fn calculate_curvature(&self, path: &[Point], idx: usize) -> f64 {
        if idx == 0 || idx >= path.len() - 1 {
            return 0.0;
        }
        
        let p0 = &path[idx - 1];
        let p1 = &path[idx];
        let p2 = &path[idx + 1];
        
        // Use Menger curvature: k = 4*Area / (a*b*c)
        // where Area is signed area of triangle, a,b,c are side lengths
        let area = Self::signed_area(p0, p1, p2);
        let a = p0.distance_to(p1);
        let b = p1.distance_to(p2);
        let c = p2.distance_to(p0);
        
        if a < 1e-6 || b < 1e-6 || c < 1e-6 {
            return 0.0;
        }
        
        4.0 * area / (a * b * c)
    }
    
    /// Calculate signed area of triangle (same as VTracer uses)
    fn signed_area(p0: &Point, p1: &Point, p2: &Point) -> f64 {
        0.5 * ((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y))
    }
    
    /// Analyze curvature along entire path
    pub fn analyze_path(&self, path: &[Point]) -> CurvatureProfile {
        if path.len() < 3 {
            return CurvatureProfile {
                curvatures: vec![0.0; path.len()],
                feature_points: vec![false; path.len()],
                adaptive_lengths: vec![self.base_segment_length; path.len()],
            };
        }
        
        // Step 1: Calculate raw curvature at each point
        let mut curvatures: Vec<f64> = (0..path.len())
            .map(|i| self.calculate_curvature(path, i))
            .collect();
        
        // Step 2: Smooth curvature profile to avoid noise
        curvatures = self.smooth_curvature(&curvatures);
        
        // Step 3: Detect feature points (high curvature = artistic detail)
        let feature_points = self.detect_features(&curvatures);
        
        // Step 4: Calculate adaptive segment lengths
        // Key insight: High curvature regions need MORE points (shorter segments)
        // Low curvature regions need FEWER points (longer segments)
        let adaptive_lengths = self.compute_adaptive_lengths(&curvatures, &feature_points);
        
        CurvatureProfile {
            curvatures,
            feature_points,
            adaptive_lengths,
        }
    }
    
    /// Smooth curvature values using moving average
    fn smooth_curvature(&self, curvatures: &[f64]) -> Vec<f64> {
        let window = self.window_size.min(curvatures.len() / 2);
        if window == 0 {
            return curvatures.to_vec();
        }
        
        curvatures
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let start = i.saturating_sub(window);
                let end = (i + window + 1).min(curvatures.len());
                let sum: f64 = curvatures[start..end].iter().map(|k| k.abs()).sum();
                let count = (end - start) as f64;
                sum / count
            })
            .collect()
    }
    
    /// Detect points that deserve artistic attention
    fn detect_features(&self, curvatures: &[f64]) -> Vec<bool> {
        curvatures
            .iter()
            .map(|&k| k.abs() > self.feature_threshold)
            .collect()
    }
    
    /// Compute adaptive segment lengths for smoothing
    /// This is the key artistic decision function
    fn compute_adaptive_lengths(
        &self,
        curvatures: &[f64],
        feature_points: &[bool],
    ) -> Vec<f64> {
        curvatures
            .iter()
            .zip(feature_points.iter())
            .map(|(&k, &is_feature)| {
                if is_feature {
                    // High curvature: use shorter segments for detail preservation
                    // Scale inversely with curvature magnitude
                    let scale = (1.0 - k.abs().min(1.0)) * 0.5 + 0.3;
                    self.base_segment_length * scale
                } else {
                    // Low curvature: use longer segments for simplification
                    // This is where the "artistry" comes from - aggressive simplification
                    let smoothness = 1.0 - k.abs().min(0.5);
                    self.base_segment_length * (1.0 + smoothness * 1.5)
                }
            })
            .collect()
    }
}

/// Enhanced path simplification that respects curvature
pub struct ArtisticSimplifier {
    analyzer: CurvatureAnalyzer,
    /// Penalty tolerance (from VTracer's original algorithm)
    penalty_tolerance: f64,
}

impl ArtisticSimplifier {
    pub fn new(base_segment_length: f64, penalty_tolerance: f64, window_size: usize, feature_threshold: f64) -> Self {
        Self {
            analyzer: CurvatureAnalyzer::new(base_segment_length, window_size, feature_threshold),
            penalty_tolerance,
        }
    }
    
    /// Enhanced simplification that preserves artistic features
    pub fn simplify_with_curvature(&self, path: &[Point]) -> Vec<Point> {
        if path.len() < 3 {
            return path.to_vec();
        }
        
        // Analyze curvature profile
        let profile = self.analyzer.analyze_path(path);
        
        let mut simplified = Vec::new();
        simplified.push(path[0]);
        
        let mut i = 1;
        while i < path.len() - 1 {
            // Check if this is a feature point - if so, NEVER remove it
            if profile.feature_points[i] {
                simplified.push(path[i]);
                i += 1;
                continue;
            }
            
            // Try to extend subpath as far as possible
            let mut j = i + 1;
            let mut max_penalty: f64 = 0.0;
            
            while j < path.len() {
                // Calculate penalty for this subpath
                let penalty = self.calculate_penalty(&path[i - 1], &path[j], &path[i..j]);
                
                // Adjust tolerance based on local curvature
                // Low curvature = more aggressive simplification allowed
                let local_curvature = profile.curvatures[i..j]
                    .iter()
                    .map(|k| k.abs())
                    .fold(0.0, f64::max);
                
                let adjusted_tolerance = if local_curvature < 0.1 {
                    self.penalty_tolerance * 2.0 // Allow more simplification in flat areas
                } else {
                    self.penalty_tolerance
                };
                
                if penalty > adjusted_tolerance {
                    break;
                }
                
                max_penalty = max_penalty.max(penalty);
                
                // Stop if we hit a feature point
                if j < path.len() && profile.feature_points[j] {
                    break;
                }
                
                j += 1;
            }
            
            // Add the endpoint of the simplified subpath
            if j > i + 1 {
                simplified.push(path[j - 1]);
                i = j - 1;
            } else {
                simplified.push(path[i]);
                i += 1;
            }
        }
        
        simplified.push(path[path.len() - 1]);
        simplified
    }
    
    /// Calculate penalty using VTracer's original formula
    fn calculate_penalty(&self, start: &Point, end: &Point, subpath: &[Point]) -> f64 {
        let base = start.distance_to(end);
        if base < 1e-6 {
            return 0.0;
        }
        
        subpath
            .iter()
            .map(|p| {
                let area = CurvatureAnalyzer::signed_area(start, p, end).abs();
                (2.0 * area) / base // h = 2*Area/base
            })
            .fold(0.0, f64::max)
    }
}

/// Example usage integrating with VTracer's pipeline
pub fn demonstrate_enhancement() {
    // Example: A simple curved path
    let path = vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.1),
        Point::new(2.0, 0.3),
        Point::new(3.0, 0.7),   // Start of curve
        Point::new(4.0, 1.5),   // High curvature
        Point::new(5.0, 2.5),   // Peak
        Point::new(6.0, 3.5),   // High curvature
        Point::new(7.0, 4.3),   // End of curve
        Point::new(8.0, 4.7),
        Point::new(9.0, 4.9),
        Point::new(10.0, 5.0),
    ];
    
    // Create analyzer with VTracer-typical parameters
            let simplifier = ArtisticSimplifier::new(8.0, 0.5, 5, 0.3);    
    // Simplify with curvature awareness
    let simplified = simplifier.simplify_with_curvature(&path);
    
    println!("Original path: {} points", path.len());
    println!("Simplified path: {} points", simplified.len());
    
    // The curve section (points 3-7) should be preserved
    // The flat sections (0-2, 8-10) should be heavily simplified
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_curvature_detection() {
        // Straight line - should have zero curvature
        let straight = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        ];
        
        let analyzer = CurvatureAnalyzer::new(8.0);
        let profile = analyzer.analyze_path(&straight);
        
        assert!(profile.curvatures[1].abs() < 0.01);
        assert!(!profile.feature_points[1]);
    }
    
    #[test]
    fn test_sharp_corner_detection() {
        // Right angle - should have high curvature
        let corner = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
        ];
        
        let analyzer = CurvatureAnalyzer::new(8.0);
        let profile = analyzer.analyze_path(&corner);
        
        assert!(profile.curvatures[1].abs() > 0.2);
        // Note: Might not be marked as feature due to smoothing window
    }
}
