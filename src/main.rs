use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    x: i32,
    y: i32,
}

impl Point {
    const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone)]
pub struct Contour {
    chain_code: Vec<u8>,
    start_point: Point,
    is_outer: bool,
    parent_id: Option<usize>,
}

impl Contour {
    // Reconstruct all points from chain code
    pub fn reconstruct_points(&self) -> Vec<Point> {
        if self.chain_code.is_empty() {
            return vec![self.start_point];
        }

        let mut points = Vec::with_capacity(self.chain_code.len() + 1);
        let mut current = self.start_point;
        points.push(current);

        for &direction in &self.chain_code {
            // 4-connected directions: 0=right, 1=up, 2=left, 3=down
            match direction {
                0 => current.x += 1, // right
                1 => current.y -= 1, // up
                2 => current.x -= 1, // left
                3 => current.y += 1, // down
                _ => continue,       // ignore invalid directions
            }
            points.push(current);
        }

        points
    }
}

// New structure for isolated pixels
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IsolatedPixel {
    pub x: i32,
    pub y: i32,
}

impl IsolatedPixel {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub fn to_point(&self) -> Point {
        Point::new(self.x, self.y)
    }
}

// Seed point for reliable flood fill operations
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SeedPoint {
    pub x: i32,
    pub y: i32,
    pub contour_id: usize, // Which contour this seed belongs to
}

impl SeedPoint {
    pub fn new(x: i32, y: i32, contour_id: usize) -> Self {
        Self { x, y, contour_id }
    }

    pub fn to_point(&self) -> Point {
        Point::new(self.x, self.y)
    }
}

// Serializable structures for compact storage
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableContour {
    pub start_x: i32,
    pub start_y: i32,
    pub is_outer: bool,
    pub parent_id: Option<usize>,
    pub packed_chain_code: Vec<u32>, // Each u32 holds up to 16 directions (2 bits each)
    pub chain_code_length: usize,    // Original length before packing
    pub should_fill: bool,           // Whether this region should be filled
    pub clockwise: bool,             // Winding order of the contour
    pub area: f64, // Signed area (positive for clockwise, negative for counter-clockwise)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FrameData {
    pub width: u32,
    pub height: u32,
    pub contours: Vec<SerializableContour>,
    pub isolated_pixels: Vec<IsolatedPixel>, // Isolated single pixels
    pub seed_points: Vec<SeedPoint>,         // Guaranteed interior points for flood fill
}

impl SerializableContour {
    // Convert from internal Contour to serializable format
    pub fn from_contour_with_analysis(
        contour: &Contour,
        should_fill: bool,
        clockwise: bool,
        area: f64,
    ) -> Self {
        let (packed_chain_code, chain_code_length) = Self::pack_chain_code(&contour.chain_code);

        Self {
            start_x: contour.start_point.x,
            start_y: contour.start_point.y,
            is_outer: contour.is_outer,
            parent_id: contour.parent_id,
            packed_chain_code,
            chain_code_length,
            should_fill,
            clockwise,
            area,
        }
    }

    // Legacy method for backward compatibility
    pub fn from_contour(contour: &Contour) -> Self {
        Self::from_contour_with_analysis(contour, contour.is_outer, false, 0.0)
    }

    // Convert back to internal Contour format
    pub fn to_contour(&self) -> Contour {
        let chain_code = Self::unpack_chain_code(&self.packed_chain_code, self.chain_code_length);

        Contour {
            start_point: Point::new(self.start_x, self.start_y),
            chain_code,
            is_outer: self.is_outer,
            parent_id: self.parent_id,
        }
    }

    // Pack chain code directions (0-3) into u32s using 2 bits per direction
    fn pack_chain_code(chain_code: &[u8]) -> (Vec<u32>, usize) {
        let original_length = chain_code.len();
        if original_length == 0 {
            return (Vec::new(), 0);
        }

        let mut packed = Vec::new();
        let mut current_u32 = 0u32;
        let mut bits_used = 0;

        for &direction in chain_code {
            // Ensure direction is valid (0-3)
            let dir = (direction & 0x03) as u32;

            // Shift the direction to its position and add to current u32
            current_u32 |= dir << bits_used;
            bits_used += 2;

            // If we've filled the u32 (16 directions * 2 bits = 32 bits), store it
            if bits_used >= 32 {
                packed.push(current_u32);
                current_u32 = 0;
                bits_used = 0;
            }
        }

        // Store any remaining bits
        if bits_used > 0 {
            packed.push(current_u32);
        }

        (packed, original_length)
    }

    // Unpack chain code from u32s back to Vec<u8>
    fn unpack_chain_code(packed: &[u32], original_length: usize) -> Vec<u8> {
        if original_length == 0 {
            return Vec::new();
        }

        let mut chain_code = Vec::with_capacity(original_length);
        let mut remaining = original_length;

        for &packed_u32 in packed {
            let mut current = packed_u32;
            let directions_in_this_u32 = std::cmp::min(16, remaining);

            for _ in 0..directions_in_this_u32 {
                let direction = (current & 0x03) as u8;
                chain_code.push(direction);
                current >>= 2;
            }

            remaining -= directions_in_this_u32;
            if remaining == 0 {
                break;
            }
        }

        chain_code
    }

    // Calculate compression statistics
    pub fn compression_stats(&self) -> (usize, usize, f64) {
        let original_bits = self.chain_code_length * 8; // 1 byte per direction originally
        let compressed_bits = self.packed_chain_code.len() * 32; // u32s
        let compression_ratio = if compressed_bits > 0 {
            original_bits as f64 / compressed_bits as f64
        } else {
            1.0
        };

        (original_bits, compressed_bits, compression_ratio)
    }
}

impl FrameData {
    pub fn from_contours_and_pixels_with_seeds(
        width: u32,
        height: u32,
        contours: &[Contour],
        isolated_pixels: &[IsolatedPixel],
        seed_points: &[SeedPoint],
        contour_metadata: &[(bool, bool, f64)], // (should_fill, clockwise, area)
    ) -> Self {
        let serializable_contours = contours
            .iter()
            .zip(contour_metadata.iter())
            .map(|(contour, &(should_fill, clockwise, area))| {
                SerializableContour::from_contour_with_analysis(
                    contour,
                    should_fill,
                    clockwise,
                    area,
                )
            })
            .collect();

        Self {
            width,
            height,
            contours: serializable_contours,
            isolated_pixels: isolated_pixels.to_vec(),
            seed_points: seed_points.to_vec(),
        }
    }

    pub fn from_contours_and_pixels(
        width: u32,
        height: u32,
        contours: &[Contour],
        isolated_pixels: &[IsolatedPixel],
    ) -> Self {
        Self::from_contours_and_pixels_with_seeds(
            width,
            height,
            contours,
            isolated_pixels,
            &[],
            &[],
        )
    }

    // Legacy method for backward compatibility
    pub fn from_contours(width: u32, height: u32, contours: &[Contour]) -> Self {
        Self::from_contours_and_pixels(width, height, contours, &[])
    }

    pub fn to_contours(&self) -> Vec<Contour> {
        self.contours.iter().map(|sc| sc.to_contour()).collect()
    }

    // Serialize to binary format using CBOR
    pub fn serialize(&self) -> Result<Vec<u8>, ciborium::ser::Error<std::io::Error>> {
        let mut buffer = Vec::new();
        ciborium::ser::into_writer(self, &mut buffer)?;
        Ok(buffer)
    }

    // Deserialize from binary format
    pub fn deserialize(data: &[u8]) -> Result<Self, ciborium::de::Error<std::io::Error>> {
        ciborium::de::from_reader(data)
    }

    // Save to file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let data = self.serialize()?;
        std::fs::write(path, data)?;
        Ok(())
    }

    // Load from file
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        Ok(Self::deserialize(&data)?)
    }

    // Print compression statistics
    pub fn print_stats(&self) {
        println!("=== Frame Serialization Statistics ===");
        println!("Image dimensions: {}x{}", self.width, self.height);
        println!("Number of contours: {}", self.contours.len());
        println!("Number of isolated pixels: {}", self.isolated_pixels.len());
        println!("Number of seed points: {}", self.seed_points.len());

        let mut total_original_bits = 0;
        let mut total_compressed_bits = 0;

        for (i, contour) in self.contours.iter().enumerate() {
            let (orig, comp, ratio) = contour.compression_stats();
            total_original_bits += orig;
            total_compressed_bits += comp;

            println!(
                "Contour {}: {} -> {} bits ({}x compression), fill: {}, clockwise: {}, area: {:.1}",
                i, orig, comp, ratio, contour.should_fill, contour.clockwise, contour.area
            );
        }

        // Calculate isolated pixels storage
        let isolated_pixels_bits = self.isolated_pixels.len() * 16; // 2 bytes per point (x, y)
        println!(
            "Isolated pixels storage: {} bits ({} bytes)",
            isolated_pixels_bits,
            (isolated_pixels_bits + 7) / 8
        );

        // Calculate seed points storage
        let seed_points_bits = self.seed_points.len() * 24; // 3 bytes per seed (x, y, contour_id)
        println!(
            "Seed points storage: {} bits ({} bytes)",
            seed_points_bits,
            (seed_points_bits + 7) / 8
        );

        let overall_ratio = if total_compressed_bits > 0 {
            total_original_bits as f64 / total_compressed_bits as f64
        } else {
            1.0
        };

        println!(
            "Overall chain code compression: {} -> {} bits ({:.2}x)",
            total_original_bits, total_compressed_bits, overall_ratio
        );

        // Estimate total serialized size
        if let Ok(serialized) = self.serialize() {
            println!(
                "Total serialized size: {} bytes ({:.2} KB)",
                serialized.len(),
                serialized.len() as f64 / 1024.0
            );
        }
    }
}

#[derive(Debug)]
pub struct ContourEncoder {
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
}

impl ContourEncoder {
    #[must_use]
    pub const fn new(target_width: u32, target_height: u32) -> Self {
        Self {
            width: 0,
            height: 0,
            target_width,
            target_height,
        }
    }

    pub fn encode_image(
        &mut self,
        image_path: &str,
    ) -> Result<(RgbImage, GrayImage, FrameData), Box<dyn std::error::Error>> {
        // Load and preprocess the image
        let img = image::open(image_path)?;
        let gray_img = img.to_luma8();

        // Resize to target dimensions
        let resized = image::imageops::resize(
            &gray_img,
            self.target_width,
            self.target_height,
            image::imageops::FilterType::Nearest,
        );

        self.width = resized.width();
        self.height = resized.height();

        // Detect if the image is inverted (more white than black pixels)
        let is_inverted = self.detect_inversion(&resized);

        println!("Image dimensions: {}x{}", self.width, self.height);
        println!("Image is inverted: {is_inverted}");

        // Convert to binary and invert if necessary
        let binary_img = self.to_binary(&resized, is_inverted);

        // Find contours using Suzuki-Abe algorithm
        let contours = self.find_contours_suzuki_abe(&binary_img);
        println!("Found {} contours", contours.len());

        // Find isolated pixels
        let isolated_pixels = self.find_isolated_pixels(&binary_img);
        println!("Found {} isolated pixels", isolated_pixels.len());

        // Compute geometric properties and seed points for flood fill
        let (contour_metadata, seed_points) = self.compute_flood_fill_data(&contours);

        // Create serializable frame data
        let frame_data = FrameData::from_contours_and_pixels_with_seeds(
            self.width,
            self.height,
            &contours,
            &isolated_pixels,
            &seed_points,
            &contour_metadata,
        );

        // Generate output images
        let overlay_img = self.create_overlay_image(&resized, &contours, &isolated_pixels);
        let contour_only_img = self.create_contour_only_image(&contours, &isolated_pixels);

        Ok((overlay_img, contour_only_img, frame_data))
    }

    // New method to find isolated pixels
    fn find_isolated_pixels(&self, binary_img: &GrayImage) -> Vec<IsolatedPixel> {
        let mut isolated_pixels = Vec::new();

        for y in 0..self.height {
            for x in 0..self.width {
                // Check if this pixel is foreground (value 1)
                if binary_img.get_pixel(x, y)[0] == 1 {
                    // Check if it's isolated (no connected foreground neighbors)
                    if self.is_isolated_pixel(binary_img, x as i32, y as i32) {
                        isolated_pixels.push(IsolatedPixel::new(x as i32, y as i32));
                    }
                }
            }
        }

        isolated_pixels
    }

    // Check if a pixel is isolated (has no 4-connected foreground neighbors)
    fn is_isolated_pixel(&self, binary_img: &GrayImage, x: i32, y: i32) -> bool {
        // 4-connected directions: right, up, left, down
        let directions = [(1, 0), (0, -1), (-1, 0), (0, 1)];

        for (dx, dy) in directions {
            let nx = x + dx;
            let ny = y + dy;

            // Check bounds
            if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                // If any neighbor is foreground, this pixel is not isolated
                if binary_img.get_pixel(nx as u32, ny as u32)[0] == 1 {
                    return false;
                }
            }
        }

        true
    }

    // Compute flood fill metadata for contours
    fn compute_flood_fill_data(
        &self,
        contours: &[Contour],
    ) -> (Vec<(bool, bool, f64)>, Vec<SeedPoint>) {
        let mut metadata = Vec::new();
        let mut seed_points = Vec::new();

        for (i, contour) in contours.iter().enumerate() {
            let points = contour.reconstruct_points();

            // Compute signed area and winding order
            let signed_area = self.compute_signed_area(&points);
            let clockwise = signed_area < 0.0;

            // Determine if this contour should be filled
            let should_fill = if contour.is_outer {
                // Outer contours should be filled if they're oriented correctly
                // In our coordinate system, clockwise outer contours are typically filled
                clockwise
            } else {
                // Holes (inner contours) should not be filled
                false
            };

            // Compute interior seed point for fillable regions
            if should_fill {
                if let Some(seed_point) = self.compute_interior_point(&points, i) {
                    seed_points.push(seed_point);
                }
            }

            metadata.push((should_fill, clockwise, signed_area.abs()));
        }

        (metadata, seed_points)
    }

    // Compute signed area using shoelace formula
    fn compute_signed_area(&self, points: &[Point]) -> f64 {
        if points.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        let n = points.len();

        for i in 0..n {
            let j = (i + 1) % n;
            let xi = points[i].x as f64;
            let yi = points[i].y as f64;
            let xj = points[j].x as f64;
            let yj = points[j].y as f64;

            area += xi * yj - xj * yi;
        }

        area / 2.0
    }

    // Compute a guaranteed interior point for flood fill
    fn compute_interior_point(&self, points: &[Point], contour_id: usize) -> Option<SeedPoint> {
        if points.len() < 3 {
            return None;
        }

        // Try centroid first
        let centroid = self.compute_centroid(points);
        if self.is_point_inside_polygon(&centroid, points) {
            return Some(SeedPoint::new(centroid.x, centroid.y, contour_id));
        }

        // If centroid fails, try the midpoint of the longest edge moved inward
        let interior_point = self.compute_safe_interior_point(points)?;
        if self.is_point_inside_polygon(&interior_point, points) {
            return Some(SeedPoint::new(
                interior_point.x,
                interior_point.y,
                contour_id,
            ));
        }

        // Last resort: try multiple points along the boundary
        self.find_any_interior_point(points, contour_id)
    }

    // Compute centroid of polygon
    fn compute_centroid(&self, points: &[Point]) -> Point {
        let sum_x: i32 = points.iter().map(|p| p.x).sum();
        let sum_y: i32 = points.iter().map(|p| p.y).sum();
        let n = points.len() as i32;

        Point::new(sum_x / n, sum_y / n)
    }

    // Compute a safe interior point by moving inward from the longest edge
    fn compute_safe_interior_point(&self, points: &[Point]) -> Option<Point> {
        if points.len() < 3 {
            return None;
        }

        // Find the longest edge
        let mut max_length_sq = 0i32;
        let mut best_edge = (0, 1);

        for i in 0..points.len() {
            let j = (i + 1) % points.len();
            let dx = points[j].x - points[i].x;
            let dy = points[j].y - points[i].y;
            let length_sq = dx * dx + dy * dy;

            if length_sq > max_length_sq {
                max_length_sq = length_sq;
                best_edge = (i, j);
            }
        }

        let (i, j) = best_edge;
        let midpoint = Point::new(
            (points[i].x + points[j].x) / 2,
            (points[i].y + points[j].y) / 2,
        );

        // Move the midpoint slightly inward (perpendicular to the edge)
        let edge_dx = points[j].x - points[i].x;
        let edge_dy = points[j].y - points[i].y;

        // Perpendicular vector (rotated 90 degrees)
        let perp_dx = -edge_dy;
        let perp_dy = edge_dx;

        // Normalize and scale the perpendicular vector
        let length = ((perp_dx * perp_dx + perp_dy * perp_dy) as f64).sqrt();
        if length > 0.0 {
            let scale = 1.0 / length; // Move 1 pixel inward
            let offset_x = (perp_dx as f64 * scale).round() as i32;
            let offset_y = (perp_dy as f64 * scale).round() as i32;

            Some(Point::new(midpoint.x + offset_x, midpoint.y + offset_y))
        } else {
            Some(midpoint)
        }
    }

    // Ray casting algorithm for point-in-polygon test
    fn is_point_inside_polygon(&self, point: &Point, polygon: &[Point]) -> bool {
        if polygon.len() < 3 {
            return false;
        }

        let mut inside = false;
        let n = polygon.len();

        let mut j = n - 1;
        for i in 0..n {
            let xi = polygon[i].x;
            let yi = polygon[i].y;
            let xj = polygon[j].x;
            let yj = polygon[j].y;

            if ((yi > point.y) != (yj > point.y))
                && (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi)
            {
                inside = !inside;
            }
            j = i;
        }

        inside
    }

    // Find any interior point as a last resort
    fn find_any_interior_point(&self, points: &[Point], contour_id: usize) -> Option<SeedPoint> {
        // Try points along edges of the bounding box
        let min_x = points.iter().map(|p| p.x).min()?;
        let max_x = points.iter().map(|p| p.x).max()?;
        let min_y = points.iter().map(|p| p.y).min()?;
        let max_y = points.iter().map(|p| p.y).max()?;

        // Try points in a grid within the bounding box
        for y in (min_y + 1)..max_y {
            for x in (min_x + 1)..max_x {
                let test_point = Point::new(x, y);
                if self.is_point_inside_polygon(&test_point, points) {
                    return Some(SeedPoint::new(x, y, contour_id));
                }
            }
        }

        None
    }

    fn detect_inversion(&self, img: &GrayImage) -> bool {
        let total_pixels = (img.width() * img.height()) as f32;
        let white_pixels = img.pixels().filter(|pixel| pixel[0] > 128).count() as f32;

        white_pixels / total_pixels > 0.5
    }

    fn to_binary(&self, img: &GrayImage, invert: bool) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let threshold = 128u8;
        ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
            let pixel = img.get_pixel(x, y)[0];
            let is_foreground = if invert {
                pixel < threshold
            } else {
                pixel > threshold
            };

            if is_foreground {
                Luma([1u8]) // Use 1 instead of 255 for Suzuki-Abe
            } else {
                Luma([0u8])
            }
        })
    }

    // 4-connected directions: 0=right, 1=up, 2=left, 3=down
    fn next_cell(&self, curr_pixel: Point, curr_dir: u8) -> (Point, u8, Option<Point>) {
        let (i, j) = (curr_pixel.y, curr_pixel.x);

        match curr_dir {
            0 => (Point::new(j, i - 1), 1, Some(Point::new(j + 1, i))), // right -> up
            1 => (Point::new(j - 1, i), 2, None),                       // up -> left
            2 => (Point::new(j, i + 1), 3, None),                       // left -> down
            3 => (Point::new(j + 1, i), 0, None),                       // down -> right
            _ => panic!("Invalid direction"),
        }
    }

    const fn is_valid_coord(&self, x: i32, y: i32) -> bool {
        x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32
    }

    fn get_pixel_value(&self, img: &[Vec<i32>], x: i32, y: i32) -> i32 {
        if self.is_valid_coord(x, y) {
            img[y as usize][x as usize]
        } else {
            0
        }
    }

    fn set_pixel_value(&self, img: &mut [Vec<i32>], x: i32, y: i32, value: i32) {
        if self.is_valid_coord(x, y) {
            img[y as usize][x as usize] = value;
        }
    }

    fn should_mark_pixel_negative(&self, save: Option<Point>, img: &[Vec<i32>]) -> bool {
        match save {
            Some(save_pt) => self.get_pixel_value(img, save_pt.x, save_pt.y) == 0,
            None => false,
        }
    }

    fn should_mark_pixel_positive(&self, save: Option<Point>, img: &[Vec<i32>]) -> bool {
        match save {
            Some(save_pt) => self.get_pixel_value(img, save_pt.x, save_pt.y) != 0,
            None => true,
        }
    }

    fn update_pixel_value(&self, img: &mut [Vec<i32>], curr: Point, nbd: i32, save: Option<Point>) {
        if self.should_mark_pixel_negative(save, img) {
            self.set_pixel_value(img, curr.x, curr.y, -nbd);
        } else if self.should_mark_pixel_positive(save, img)
            && self.get_pixel_value(img, curr.x, curr.y) == 1
        {
            self.set_pixel_value(img, curr.x, curr.y, nbd);
        }
    }

    fn border_follow(
        &self,
        img: &mut [Vec<i32>],
        start: Point,
        prev: Point,
        mut direction: u8,
        nbd: i32,
    ) -> Vec<Point> {
        let mut curr = start;
        let mut exam = prev;
        let save2 = exam;
        let mut contour = vec![curr];
        let mut save: Option<Point> = None;

        // Find first non-zero pixel
        while self.get_pixel_value(img, exam.x, exam.y) == 0 {
            let (new_exam, new_dir, save_pixel) = self.next_cell(curr, direction);
            exam = new_exam;
            direction = new_dir;
            if save_pixel.is_some() {
                save = save_pixel;
            }
            if save2.x == exam.x && save2.y == exam.y {
                self.set_pixel_value(img, curr.x, curr.y, -nbd);
                return contour;
            }
        }

        // Update pixel values
        self.update_pixel_value(img, curr, nbd, save);
        save = None;

        let mut prev = curr;
        curr = exam;
        contour.push(curr);

        direction = if direction >= 2 {
            direction - 2
        } else {
            direction + 2
        };

        let mut flag = false;
        let start_next = curr;

        loop {
            if curr.x == start_next.x
                && curr.y == start_next.y
                && prev.x == start.x
                && prev.y == start.y
                && flag
            {
                break;
            }

            flag = true;
            let (new_exam, new_dir, save_pixel) = self.next_cell(curr, direction);
            exam = new_exam;
            direction = new_dir;
            if save_pixel.is_some() {
                save = save_pixel;
            }
            while self.get_pixel_value(img, exam.x, exam.y) == 0 {
                let (new_exam, new_dir, save_pixel) = self.next_cell(curr, direction);
                exam = new_exam;
                direction = new_dir;
                if save_pixel.is_some() {
                    save = save_pixel;
                }
            }
            // Update pixel values
            self.update_pixel_value(img, curr, nbd, save);
            save = None;
            prev = curr;
            curr = exam;
            contour.push(curr);
            direction = if direction >= 2 {
                direction - 2
            } else {
                direction + 2
            };
        }

        contour
    }

    fn find_contours_suzuki_abe(&self, binary_img: &GrayImage) -> Vec<Contour> {
        let rows = self.height as usize;
        let cols = self.width as usize;

        // Convert to working image format
        let mut img = vec![vec![0i32; cols]; rows];
        for y in 0..rows {
            for x in 0..cols {
                img[y][x] = i32::from(binary_img.get_pixel(x as u32, y as u32)[0]);
            }
        }

        let mut lnbd;
        let mut nbd = 1i32;
        let mut contours = Vec::new();
        let mut parents = vec![-1i32]; // parents[i] = parent border number of border i
        let mut border_types = vec![0u8]; // 0 = hole, 1 = outer
        let mut border_to_contour: Vec<Option<usize>> = vec![None]; // Map border number to contour index

        for i in 1..rows - 1 {
            lnbd = 1;
            for j in 1..cols - 1 {
                let curr_val = img[i][j];
                let left_val = img[i][j - 1];
                let right_val = img[i][j + 1];

                // Check for outer boundary (object border)
                if curr_val == 1 && left_val == 0 {
                    nbd += 1;
                    let direction = 2; // left
                    parents.push(lnbd);

                    let contour_points = self.border_follow(
                        &mut img,
                        Point::new(j as i32, i as i32),
                        Point::new((j - 1) as i32, i as i32),
                        direction,
                        nbd,
                    );

                    if contour_points.len() > 2 {
                        let parent_id =
                            self.find_parent_contour_id(nbd, &parents, &border_to_contour);
                        let contour = self.create_contour_from_points_with_parent(
                            &contour_points,
                            true,
                            parent_id,
                        );
                        let contour_idx = contours.len();
                        contours.push(contour);

                        // Update mapping
                        if border_to_contour.len() <= nbd as usize {
                            border_to_contour.resize(nbd as usize + 1, None);
                        }
                        border_to_contour[nbd as usize] = Some(contour_idx);
                    } else {
                        border_to_contour.push(None);
                    }

                    border_types.push(1);
                    self.update_lnbd_for_outer_boundary(
                        &border_types,
                        &parents,
                        nbd,
                        &mut lnbd,
                        &img,
                        i,
                        j,
                    );
                }
                // Check for hole boundary
                else if curr_val >= 1 && right_val == 0 {
                    nbd += 1;
                    let direction = 0; // right
                    if curr_val > 1 {
                        lnbd = curr_val;
                    }
                    parents.push(lnbd);

                    let contour_points = self.border_follow(
                        &mut img,
                        Point::new(j as i32, i as i32),
                        Point::new((j + 1) as i32, i as i32),
                        direction,
                        nbd,
                    );

                    if contour_points.len() > 2 {
                        let parent_id =
                            self.find_parent_contour_id(nbd, &parents, &border_to_contour);
                        let contour = self.create_contour_from_points_with_parent(
                            &contour_points,
                            false,
                            parent_id,
                        );
                        let contour_idx = contours.len();
                        contours.push(contour);

                        // Update mapping
                        if border_to_contour.len() <= nbd as usize {
                            border_to_contour.resize(nbd as usize + 1, None);
                        }
                        border_to_contour[nbd as usize] = Some(contour_idx);
                    } else {
                        border_to_contour.push(None);
                    }

                    border_types.push(0);
                    self.update_lnbd_for_hole_boundary(
                        &border_types,
                        &parents,
                        nbd,
                        &mut lnbd,
                        &img,
                        i,
                        j,
                    );
                }

                // Update LNBD
                let curr_pixel_val = img[i][j];
                if curr_pixel_val != 1 {
                    lnbd = curr_pixel_val.abs();
                }
            }
        }

        contours
    }

    fn find_parent_contour_id(
        &self,
        nbd: i32,
        parents: &[i32],
        border_to_contour: &[Option<usize>],
    ) -> Option<usize> {
        // Get the parent border number for this border
        if let Some(&parent_border) = parents.get(nbd as usize) {
            if parent_border <= 1 {
                // No parent or background parent
                return None;
            }

            // Look up the contour index for the parent border
            if let Some(&Some(parent_contour_idx)) = border_to_contour.get(parent_border as usize) {
                return Some(parent_contour_idx);
            }
        }

        None
    }

    fn create_contour_from_points_with_parent(
        &self,
        points: &[Point],
        is_outer: bool,
        parent_id: Option<usize>,
    ) -> Contour {
        let chain_code = self.points_to_chain_code(points);

        Contour {
            start_point: points[0],
            chain_code,
            is_outer,
            parent_id,
        }
    }

    fn update_lnbd_for_outer_boundary(
        &self,
        border_types: &[u8],
        _parents: &[i32],
        nbd: i32,
        lnbd: &mut i32,
        img: &[Vec<i32>],
        i: usize,
        j: usize,
    ) {
        if border_types.get(nbd as usize - 2).is_none_or(|&bt| bt != 1) {
            let curr_pixel_val = img[i][j];
            if curr_pixel_val != 1 {
                *lnbd = curr_pixel_val.abs();
            }
        }
    }

    fn update_lnbd_for_hole_boundary(
        &self,
        border_types: &[u8],
        _parents: &[i32],
        nbd: i32,
        lnbd: &mut i32,
        img: &[Vec<i32>],
        i: usize,
        j: usize,
    ) {
        if border_types.get(nbd as usize - 2).is_none_or(|&bt| bt != 0) {
            let curr_pixel_val = img[i][j];
            if curr_pixel_val != 1 {
                *lnbd = curr_pixel_val.abs();
            }
        }
    }

    fn points_to_chain_code(&self, points: &[Point]) -> Vec<u8> {
        let mut chain_code = Vec::new();

        for i in 0..points.len() {
            let curr = points[i];
            let next = points[(i + 1) % points.len()];

            let dx = next.x - curr.x;
            let dy = next.y - curr.y;

            // 4-connected chain code
            let code = match (dx, dy) {
                (1, 0) => 0,   // right
                (0, -1) => 1,  // up
                (-1, 0) => 2,  // left
                (0, 1) => 3,   // down
                _ => continue, // skip diagonal or invalid moves
            };

            chain_code.push(code);
        }

        chain_code
    }

    fn create_overlay_image(
        &self,
        base_img: &GrayImage,
        contours: &[Contour],
        isolated_pixels: &[IsolatedPixel],
    ) -> RgbImage {
        let mut overlay = RgbImage::new(self.width, self.height);

        // Copy base image as grayscale
        for (x, y, pixel) in base_img.enumerate_pixels() {
            let gray_val = pixel[0];
            overlay.put_pixel(x, y, Rgb([gray_val, gray_val, gray_val]));
        }

        // Draw contours reconstructed from chain codes
        for contour in contours {
            let reconstructed_points = contour.reconstruct_points();

            let color = if contour.is_outer {
                Rgb([255, 0, 0]) // red for outer boundaries
            } else {
                Rgb([0, 0, 255]) // blue for holes
            };

            for point in &reconstructed_points {
                if self.is_valid_coord(point.x, point.y) {
                    overlay.put_pixel(point.x as u32, point.y as u32, color);
                }
            }
        }

        // Draw isolated pixels in green
        for pixel in isolated_pixels {
            if self.is_valid_coord(pixel.x, pixel.y) {
                overlay.put_pixel(pixel.x as u32, pixel.y as u32, Rgb([0, 255, 0]));
            }
        }

        overlay
    }

    fn create_contour_only_image(
        &self,
        contours: &[Contour],
        isolated_pixels: &[IsolatedPixel],
    ) -> GrayImage {
        let mut contour_img = GrayImage::new(self.width, self.height);

        // Initialize with black background
        for pixel in contour_img.pixels_mut() {
            *pixel = Luma([0u8]);
        }

        // Draw contours reconstructed from chain codes in white
        for contour in contours {
            let reconstructed_points = contour.reconstruct_points();

            for point in &reconstructed_points {
                if self.is_valid_coord(point.x, point.y) {
                    contour_img.put_pixel(point.x as u32, point.y as u32, Luma([255u8]));
                }
            }
        }

        // Draw isolated pixels in white as well
        for pixel in isolated_pixels {
            if self.is_valid_coord(pixel.x, pixel.y) {
                contour_img.put_pixel(pixel.x as u32, pixel.y as u32, Luma([255u8]));
            }
        }

        contour_img
    }

    pub fn print_chain_codes(
        &self,
        contours: &[Contour],
        isolated_pixels: &[IsolatedPixel],
        seed_points: &[SeedPoint],
    ) {
        println!("=== Contour Analysis with Chain Code Encoding ===");
        println!();

        for (i, contour) in contours.iter().enumerate() {
            let boundary_type = if contour.is_outer { "Outer" } else { "Hole" };
            let parent_info = match contour.parent_id {
                Some(pid) => format!("Parent: {pid}"),
                None => "Root".to_string(),
            };

            println!("Contour {i}: {boundary_type} boundary, {parent_info}");
            println!(
                "  Start Point: ({}, {})",
                contour.start_point.x, contour.start_point.y
            );

            // Chain code stats
            let reconstructed_points = contour.reconstruct_points();
            println!("  Reconstructed points: {}", reconstructed_points.len());
            println!("  Chain code length: {}", contour.chain_code.len());
            println!(
                "  Chain code: {:?}",
                if contour.chain_code.len() > 20 {
                    format!("{:?}...", &contour.chain_code[..20])
                } else {
                    format!("{:?}", contour.chain_code)
                }
            );

            // Storage analysis
            let coordinate_storage_bits = reconstructed_points.len() * 16; // 2 bytes per point
            let chain_code_storage_bits = 16 + (contour.chain_code.len() * 2); // start point + 2 bits per direction

            let compression_ratio = if chain_code_storage_bits > 0 {
                coordinate_storage_bits as f64 / chain_code_storage_bits as f64
            } else {
                1.0
            };

            println!("  Storage Analysis:");
            println!(
                "    Coordinates: {} bits ({} bytes)",
                coordinate_storage_bits,
                coordinate_storage_bits / 8
            );
            println!(
                "    Chain codes: {} bits ({} bytes)",
                chain_code_storage_bits,
                (chain_code_storage_bits + 7) / 8
            );
            println!("    Compression ratio: {:.2}x", compression_ratio);
            println!("  ---");
        }

        // Isolated pixels analysis
        if !isolated_pixels.is_empty() {
            println!("=== Isolated Pixels Analysis ===");
            println!("Number of isolated pixels: {}", isolated_pixels.len());
            let isolated_storage_bits = isolated_pixels.len() * 16; // 2 bytes per point (x, y)
            println!(
                "Storage: {} bits ({} bytes)",
                isolated_storage_bits,
                (isolated_storage_bits + 7) / 8
            );

            // Show first few isolated pixels as examples
            let num_to_show = std::cmp::min(10, isolated_pixels.len());
            print!("Sample isolated pixels: ");
            for (i, pixel) in isolated_pixels.iter().take(num_to_show).enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("({}, {})", pixel.x, pixel.y);
            }
            if isolated_pixels.len() > num_to_show {
                print!("...");
            }
            println!();
            println!("  ---");
        }

        // Seed points analysis
        if !seed_points.is_empty() {
            println!("=== Flood Fill Seed Points Analysis ===");
            println!("Number of seed points: {}", seed_points.len());
            let seed_storage_bits = seed_points.len() * 24; // 3 bytes per seed (x, y, contour_id)
            println!(
                "Storage: {} bits ({} bytes)",
                seed_storage_bits,
                (seed_storage_bits + 7) / 8
            );

            println!("Seed points for reliable flood fill:");
            for seed in seed_points {
                println!(
                    "  Contour {}: seed at ({}, {})",
                    seed.contour_id, seed.x, seed.y
                );
            }
            println!("  ---");
        }

        // Overall statistics
        let total_reconstructed_points: usize =
            contours.iter().map(|c| c.reconstruct_points().len()).sum();
        let total_chain_code_length: usize = contours.iter().map(|c| c.chain_code.len()).sum();

        let total_coordinate_bits = total_reconstructed_points * 16;
        let total_chain_code_bits = contours.len() * 16 + total_chain_code_length * 2; // start points + chain codes
        let total_isolated_bits = isolated_pixels.len() * 16;

        let overall_compression_ratio = if total_chain_code_bits > 0 {
            total_coordinate_bits as f64 / total_chain_code_bits as f64
        } else {
            1.0
        };

        println!("=== Overall Statistics ===");
        println!("Total contours: {}", contours.len());
        println!("Total isolated pixels: {}", isolated_pixels.len());
        println!("Total reconstructed points: {}", total_reconstructed_points);
        println!("Total chain code length: {}", total_chain_code_length);
        println!(
            "Contour storage: {} bits -> {} bits (compression: {:.2}x)",
            total_coordinate_bits, total_chain_code_bits, overall_compression_ratio
        );
        println!("Isolated pixels storage: {} bits", total_isolated_bits);
        println!(
            "Total feature storage: {} bits ({:.1} KB)",
            total_chain_code_bits + total_isolated_bits,
            (total_chain_code_bits + total_isolated_bits) as f64 / 8192.0
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <image_path>", args[0]);
        std::process::exit(1);
    }

    let image_path = &args[1];
    let mut encoder = ContourEncoder::new(180, 135);

    match encoder.encode_image(image_path) {
        Ok((overlay_img, contour_only_img, frame_data)) => {
            // Save output images
            let base_name = Path::new(image_path).file_stem().unwrap().to_str().unwrap();

            let overlay_path = format!("{base_name}_overlay.png");
            let contour_path = format!("{base_name}_contour.png");
            let data_path = format!("{base_name}_frame.cbor");

            overlay_img.save(&overlay_path)?;
            contour_only_img.save(&contour_path)?;
            frame_data.save_to_file(&data_path)?;

            println!("Saved overlay image to: {overlay_path}");
            println!("Saved contour image to: {contour_path}");
            println!("Saved frame data to: {data_path}");

            // Print statistics
            frame_data.print_stats();

            // Print chain codes for analysis
            let contours = frame_data.to_contours();
            encoder.print_chain_codes(
                &contours,
                &frame_data.isolated_pixels,
                &frame_data.seed_points,
            );
        }
        Err(e) => {
            eprintln!("Error processing image: {e}");
            std::process::exit(1);
        }
    }

    Ok(())
}
