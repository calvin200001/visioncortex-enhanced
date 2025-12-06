

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureType {
    /// Sharp corner (high curvature, abrupt angle change) - like corners of "5", "7"
    Corner,
    /// Smooth curve (high curvature, gradual) - like bowls of "3", "9"  
    Curve,
    /// Straight line (low curvature) - can be simplified aggressively
    Straight,
}

#[derive(Debug)]
pub struct CurvatureProfile {
    /// Curvature value at each point
    pub curvatures: Vec<f64>,
    /// Classify each point's feature type
    pub feature_types: Vec<FeatureType>,
    /// Whether each point is a critical feature (for backwards compatibility)
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
    
    /// Analyze curvature along entire path
    pub fn analyze_path(&self, path: &[Point]) -> CurvatureProfile {
        if path.len() < 3 {
            return CurvatureProfile {
                curvatures: vec![0.0; path.len()],
                feature_types: vec![FeatureType::Straight; path.len()],
                feature_points: vec![false; path.len()],
                adaptive_lengths: vec![self.base_segment_length; path.len()],
            };
        }
        
        // Step 1: Calculate raw curvature
        let mut curvatures: Vec<f64> = (0..path.len())
            .map(|i| self.calculate_curvature(path, i))
            .collect();
        
        // Step 2: Smooth curvature
        curvatures = self.smooth_curvature(&curvatures);
        
        // Step 3: Classify feature types (NEW)
        let feature_types = self.classify_features(&curvatures, path);
        
        // Step 4: Mark critical features
        let feature_points = feature_types.iter()
            .map(|ft| *ft != FeatureType::Straight)
            .collect();
        
        // Step 5: Adaptive segment lengths
        let adaptive_lengths = self.compute_adaptive_lengths_by_type(&curvatures, &feature_types);
        
        CurvatureProfile {
            curvatures,
            feature_types,
            feature_points,
            adaptive_lengths,
        }
    }
    
    /// Classify each point as Corner, Curve, or Straight
    fn classify_features(&self, curvatures: &[f64], path: &[Point]) -> Vec<FeatureType> {
        let len = curvatures.len();
        let mut types = vec![FeatureType::Straight; len];
        
        let window = 3;  // Look ±3 points
        
        for i in 0..len {
            let k = curvatures[i].abs();
            
            if k < self.feature_threshold * 0.5 {
                // Very low curvature = straight line
                types[i] = FeatureType::Straight;
            } else {
                // High curvature - need to distinguish corner vs curve
                // Look at a WIDER window for stable classification
                let mut local_curvatures = Vec::new();
                for offset in -(window as i32)..=(window as i32) {
                    let idx = ((i as i32 + offset).rem_euclid(len as i32)) as usize;
                    local_curvatures.push(curvatures[idx].abs());
                }

                // Calculate variance - high variance = corner, low variance = smooth curve
                let mean = local_curvatures.iter().sum::<f64>() / local_curvatures.len() as f64;
                let variance = local_curvatures.iter()
                    .map(|&c| (c - mean).powi(2))
                    .sum::<f64>() / local_curvatures.len() as f64;
                
                if variance > 0.1 {  // High variance = abrupt change = corner
                    types[i] = FeatureType::Corner;
                } else {  // High curvature but smooth = curve
                    types[i] = FeatureType::Curve;
                }
            }
        }
        
        types
    }
    
    /// Compute adaptive lengths based on feature classification
    fn compute_adaptive_lengths_by_type(
        &self,
        curvatures: &[f64],
        feature_types: &[FeatureType],
    ) -> Vec<f64> {
        curvatures
            .iter()
            .zip(feature_types.iter())
            .map(|(&k, &ft)| {
                match ft {
                    FeatureType::Corner => {
                        // Corners need very short segments to preserve sharpness
                        self.base_segment_length * 0.3
                    }
                    FeatureType::Curve => {
                        // Curves need moderate segments to capture smoothness
                        let density = 1.0 - (k.abs().min(1.0) * 0.5);
                        self.base_segment_length * density.clamp(0.4, 0.8)
                    }
                    FeatureType::Straight => {
                        // Straight lines can be simplified aggressively
                        self.base_segment_length * 2.0
                    }
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
    
    /// Enhanced simplification that preserves corners and curves differently
    pub fn simplify_with_curvature(&self, path: &[Point]) -> Vec<Point> {
        if path.len() < 3 {
            return path.to_vec();
        }
        
        let profile = self.analyzer.analyze_path(path);
        
        let mut simplified = Vec::new();
        simplified.push(path[0]);
        
        let mut i = 1;
        while i < path.len() - 1 {
            let feature_type = profile.feature_types[i];
            
            // NEVER remove corners or curves
            if feature_type == FeatureType::Corner || feature_type == FeatureType::Curve {
                simplified.push(path[i]);
                i += 1;
                continue;
            }
            
            // For straight sections, try to extend as far as possible
            let mut j = i + 1;
            
            while j < path.len() {
                // Stop if we hit a corner or curve
                if j < profile.feature_types.len() {
                    let next_type = profile.feature_types[j];
                    if next_type == FeatureType::Corner || next_type == FeatureType::Curve {
                        break;
                    }
                }
                
                // Calculate penalty for this segment
                let penalty = self.calculate_penalty(&path[i - 1], &path[j], &path[i..j]);
                
                // Use adjusted tolerance based on local curvature
                let local_curvature = profile.curvatures[i..j]
                    .iter()
                    .map(|k| k.abs())
                    .fold(0.0, f64::max);
                
                let adjusted_tolerance = if local_curvature < 0.05 {
                    // Very straight - allow aggressive simplification
                    self.penalty_tolerance * 3.0
                } else {
                    self.penalty_tolerance
                };
                
                if penalty > adjusted_tolerance {
                    break;
                }
                
                j += 1;
            }
            
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
