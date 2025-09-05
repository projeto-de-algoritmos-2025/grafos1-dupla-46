use image::{GrayImage, ImageBuffer, Luma};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const EPSILON: f64 = 3.0;
const EPSILON_SQUARED: f64 = EPSILON * EPSILON;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    x: i32,
    y: i32,
}

impl Point {
    const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    const fn to_f64(&self) -> Point2D {
        Point2D {
            x: self.x as f64,
            y: self.y as f64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    x: f64,
    y: f64,
}

impl Point2D {
    #[allow(dead_code)]
    const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn to_i32(&self) -> Point {
        Point::new(self.x.round() as i32, self.y.round() as i32)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Vector2D {
    x: f64,
    y: f64,
}

impl Vector2D {
    fn new(p1: Point2D, p2: Point2D) -> Self {
        Self {
            x: p2.x - p1.x,
            y: p2.y - p1.y,
        }
    }

    fn abs2(&self) -> f64 {
        self.x.mul_add(self.x, self.y * self.y)
    }

    #[allow(dead_code)]
    fn dot(&self, other: &Self) -> f64 {
        self.x.mul_add(other.x, self.y * other.y)
    }
}

// Packed chain code structure for efficient storage
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PackedChainCode {
    pub data: Vec<u32>, // Each u32 stores up to 16 directions (2 bits each)
    pub length: usize,  // Actual number of direction codes
}

impl PackedChainCode {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            length: 0,
        }
    }

    pub fn from_directions(directions: &[u8]) -> Self {
        let mut packed = Self::new();
        packed.pack_directions(directions);
        packed
    }

    fn pack_directions(&mut self, directions: &[u8]) {
        self.length = directions.len();
        self.data.clear();

        if directions.is_empty() {
            return;
        }

        // Calculate how many u32s we need (16 directions per u32)
        let num_u32s = (directions.len() + 15) / 16;
        self.data.reserve(num_u32s);

        let mut current_u32 = 0u32;
        let mut bit_position = 0;

        for &direction in directions {
            // Pack 2 bits for each direction
            current_u32 |= (direction as u32 & 0b11) << bit_position;
            bit_position += 2;

            // If we've filled 16 directions (32 bits), store and start new u32
            if bit_position >= 32 {
                self.data.push(current_u32);
                current_u32 = 0;
                bit_position = 0;
            }
        }

        // Store the last partially filled u32 if needed
        if bit_position > 0 {
            self.data.push(current_u32);
        }
    }

    #[allow(dead_code)]
    pub fn unpack_directions(&self) -> Vec<u8> {
        let mut directions = Vec::with_capacity(self.length);

        for (chunk_idx, &packed_data) in self.data.iter().enumerate() {
            for bit_pos in (0..32).step_by(2) {
                let direction_idx = chunk_idx * 16 + bit_pos / 2;
                if direction_idx >= self.length {
                    break;
                }

                let direction = ((packed_data >> bit_pos) & 0b11) as u8;
                directions.push(direction);
            }
        }

        directions
    }

    pub fn bits_used(&self) -> usize {
        self.data.len() * 32
    }

    pub fn compression_ratio(&self) -> f64 {
        if self.length == 0 {
            return 1.0;
        }
        (self.length * 8) as f64 / self.bits_used() as f64
    }
}

// Serializable structures for pot encoding
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct SerializablePoint {
    pub x: i32,
    pub y: i32,
}

impl From<Point> for SerializablePoint {
    fn from(point: Point) -> Self {
        Self {
            x: point.x,
            y: point.y,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableContour {
    pub chain_code: PackedChainCode,
    pub start_point: SerializablePoint,
    pub is_outer: bool,
    pub parent_id: Option<usize>,
    pub point_count: usize, // Original number of points before simplification
    pub simplified_count: usize, // Number of points after simplification
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ImageContours {
    pub filename: String,
    pub width: u32,
    pub height: u32,
    pub target_width: u32,
    pub target_height: u32,
    pub was_inverted: bool,
    pub contours: Vec<SerializableContour>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ContourDatabase {
    pub images: Vec<ImageContours>,
    pub encoder_version: String,
    pub epsilon: f64,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct Contour {
    points: Vec<Point>,
    simplified_points: Vec<Point>,
    chain_code: PackedChainCode,
    start_point: Point,
    is_outer: bool,
    parent_id: Option<usize>,
}

impl Contour {
    fn to_serializable(&self) -> SerializableContour {
        SerializableContour {
            chain_code: self.chain_code.clone(),
            start_point: self.start_point.into(),
            is_outer: self.is_outer,
            parent_id: self.parent_id,
            point_count: self.points.len(),
            simplified_count: self.simplified_points.len(),
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

    // Ramer-Douglas-Peucker algorithm implementation
    fn distance_point_to_line_squared(
        &self,
        point: Point2D,
        line_start: Point2D,
        line_end: Point2D,
    ) -> f64 {
        let line_vec = Vector2D::new(line_start, line_end);
        let line_length_squared = line_vec.abs2();

        if line_length_squared == 0.0 {
            // Line is actually a point, return distance to that point
            let vec_to_point = Vector2D::new(line_start, point);
            return vec_to_point.abs2();
        }

        // Using the 2D optimized formula from the C++ implementation
        let dist = line_vec.x.mul_add(
            line_start.y - point.y,
            -(line_vec.y * (line_start.x - point.x)),
        );
        let dist_squared = dist * dist;

        // Normalize by line length squared for proper distance
        dist_squared / line_length_squared
    }

    fn find_most_distant_point(
        &self,
        points: &[Point2D],
        start_idx: usize,
        end_idx: usize,
    ) -> (f64, usize) {
        assert!(
            start_idx < end_idx,
            "Start index must be smaller than end index"
        );
        assert!(
            end_idx < points.len(),
            "End index is larger than the number of points"
        );
        assert!(points.len() >= 2, "At least two points needed");

        let line_start = points[start_idx];
        let line_end = points[end_idx];

        let mut max_dist_squared = 0.0;
        let mut max_dist_index = start_idx;

        for i in (start_idx + 1)..end_idx {
            let dist_squared = self.distance_point_to_line_squared(points[i], line_start, line_end);

            if dist_squared > max_dist_squared {
                max_dist_squared = dist_squared;
                max_dist_index = i;
            }
        }

        (max_dist_squared, max_dist_index)
    }

    fn ramer_douglas_peucker_recursive(
        &self,
        points: &[Point2D],
        start_idx: usize,
        end_idx: usize,
        epsilon_squared: f64,
        indices_to_keep: &mut Vec<usize>,
    ) {
        assert!(
            start_idx < end_idx,
            "Start index must be smaller than end index"
        );
        assert!(
            end_idx < points.len(),
            "End index is larger than the number of points"
        );
        assert!(
            epsilon_squared >= 0.0,
            "epsilon_squared must be non-negative"
        );

        let (max_dist_squared, max_dist_index) =
            self.find_most_distant_point(points, start_idx, end_idx);

        if max_dist_squared > epsilon_squared {
            // Point is far enough, recursively simplify both segments
            self.ramer_douglas_peucker_recursive(
                points,
                start_idx,
                max_dist_index,
                epsilon_squared,
                indices_to_keep,
            );
            self.ramer_douglas_peucker_recursive(
                points,
                max_dist_index,
                end_idx,
                epsilon_squared,
                indices_to_keep,
            );
        } else {
            // All points between start and end are close enough, keep only the end point
            indices_to_keep.push(end_idx);
        }
    }

    fn simplify_contour(&self, points: &[Point]) -> Vec<Point> {
        if points.len() <= 2 {
            return points.to_vec();
        }

        // Convert to f64 points for processing
        let f64_points: Vec<Point2D> = points.iter().map(Point::to_f64).collect();
        let epsilon_squared = EPSILON_SQUARED;

        let mut indices_to_keep = vec![0]; // Always keep the first point

        // For closed contours, we need to handle the wraparound
        let is_closed = points.first() == points.last();
        let end_idx = if is_closed && points.len() > 2 {
            points.len() - 2 // Skip the duplicate last point
        } else {
            points.len() - 1
        };

        if end_idx > 0 {
            self.ramer_douglas_peucker_recursive(
                &f64_points,
                0,
                end_idx,
                epsilon_squared,
                &mut indices_to_keep,
            );
        }

        // Sort indices and remove duplicates
        indices_to_keep.sort_unstable();
        indices_to_keep.dedup();

        // Convert back to i32 points
        let mut simplified: Vec<Point> = indices_to_keep
            .iter()
            .map(|&idx| f64_points[idx].to_i32())
            .collect();

        // For closed contours, ensure the last point equals the first
        if is_closed && !simplified.is_empty() && simplified.first() != simplified.last() {
            simplified.push(simplified[0]);
        }

        simplified
    }

    pub fn encode_image(
        &mut self,
        image_path: &Path,
    ) -> Result<ImageContours, Box<dyn std::error::Error>> {
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

        println!(
            "Processing: {} ({}x{}, inverted: {})",
            image_path.display(),
            self.width,
            self.height,
            is_inverted
        );

        // Convert to binary and invert if necessary
        let binary_img = self.to_binary(&resized, is_inverted);

        // Find contours using Suzuki-Abe algorithm
        let contours = self.find_contours_suzuki_abe(&binary_img);

        println!("  Found {} contours", contours.len());

        // Convert to serializable format
        let serializable_contours: Vec<SerializableContour> =
            contours.iter().map(|c| c.to_serializable()).collect();

        let filename = image_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(ImageContours {
            filename,
            width: self.width,
            height: self.height,
            target_width: self.target_width,
            target_height: self.target_height,
            was_inverted: is_inverted,
            contours: serializable_contours,
        })
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

        let mut lnbd = 1i32;
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
        let simplified_points = self.simplify_contour(points);
        let chain_code = self.points_to_chain_code(points);

        Contour {
            start_point: points[0],
            points: points.to_owned(),
            simplified_points,
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

    fn points_to_chain_code(&self, points: &[Point]) -> PackedChainCode {
        let mut directions = Vec::new();

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

            directions.push(code);
        }

        PackedChainCode::from_directions(&directions)
    }

    pub fn encode_directory(
        &mut self,
        directory_path: &str,
    ) -> Result<ContourDatabase, Box<dyn std::error::Error>> {
        let dir_path = Path::new(directory_path);
        if !dir_path.is_dir() {
            return Err(format!("Path '{}' is not a directory", directory_path).into());
        }

        println!("Scanning directory: {}", directory_path);

        // Find all PNG files in the directory
        let mut png_files = Vec::new();
        for entry in fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension.to_ascii_lowercase() == "png" {
                        png_files.push(path);
                    }
                }
            }
        }

        if png_files.is_empty() {
            return Err("No PNG files found in the directory".into());
        }

        png_files.sort(); // Sort for consistent ordering
        println!("Found {} PNG files", png_files.len());

        // Process each PNG file
        let mut images = Vec::new();
        let mut total_contours = 0;
        let mut total_original_points = 0;
        let mut total_simplified_points = 0;
        let mut total_chain_code_bits = 0;
        let mut total_uncompressed_chain_bits = 0;

        for png_file in &png_files {
            match self.encode_image(png_file) {
                Ok(image_contours) => {
                    // Collect statistics
                    total_contours += image_contours.contours.len();
                    for contour in &image_contours.contours {
                        total_original_points += contour.point_count;
                        total_simplified_points += contour.simplified_count;
                        total_chain_code_bits += contour.chain_code.bits_used();
                        total_uncompressed_chain_bits += contour.chain_code.length * 8; // 8 bits per u8
                    }

                    println!("  {} contours processed", image_contours.contours.len());
                    images.push(image_contours);
                }
                Err(e) => {
                    eprintln!("Error processing {}: {}", png_file.display(), e);
                    continue;
                }
            }
        }

        println!("\n=== Overall Statistics ===");
        println!("Total images processed: {}", images.len());
        println!("Total contours: {}", total_contours);
        println!(
            "Total points: {} -> {} (reduction: {:.2}x)",
            total_original_points,
            total_simplified_points,
            if total_simplified_points > 0 {
                total_original_points as f64 / total_simplified_points as f64
            } else {
                1.0
            }
        );

        let chain_compression_ratio = if total_chain_code_bits > 0 {
            total_uncompressed_chain_bits as f64 / total_chain_code_bits as f64
        } else {
            1.0
        };

        println!(
            "Chain code compression: {} -> {} bits ({:.2}x reduction)",
            total_uncompressed_chain_bits, total_chain_code_bits, chain_compression_ratio
        );

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let database = ContourDatabase {
            images,
            encoder_version: "1.0".to_string(),
            epsilon: EPSILON,
            created_at: timestamp,
        };

        Ok(database)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <directory_path>", args[0]);
        eprintln!("This will process all PNG files in the directory and create a .pot file");
        std::process::exit(1);
    }

    let directory_path = &args[1];
    let mut encoder = ContourEncoder::new(180, 135);

    println!("=== Contour Encoder with Pot Serialization ===");
    println!("Target resolution: {}x{}", 180, 135);
    println!("Douglas-Peucker epsilon: {}", EPSILON);
    println!();

    match encoder.encode_directory(directory_path) {
        Ok(database) => {
            // Serialize to pot format
            let serialized = pot::to_vec(&database)?;

            // Create output filename
            let dir_path = Path::new(directory_path);
            let dir_name = dir_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("contours");
            let output_filename = format!("{}_contours.pot", dir_name);

            // Save to file
            fs::write(&output_filename, &serialized)?;

            println!("\n=== Serialization Complete ===");
            println!("Output file: {}", output_filename);
            println!("File size: {:.1} KB", serialized.len() as f64 / 1024.0);
            println!(
                "Compression ratio: {:.2}x",
                (database.images.len() * 180 * 135 * 3) as f64 / serialized.len() as f64
            );

            // Print details about chain code efficiency
            let total_packed_chain_bits: usize = database
                .images
                .iter()
                .flat_map(|img| &img.contours)
                .map(|c| c.chain_code.bits_used())
                .sum();

            let total_original_chain_bits: usize = database
                .images
                .iter()
                .flat_map(|img| &img.contours)
                .map(|c| c.chain_code.length * 8) // 8 bits if stored as Vec<u8>
                .sum();

            println!(
                "Total chain codes: {} directions",
                database
                    .images
                    .iter()
                    .flat_map(|img| &img.contours)
                    .map(|c| c.chain_code.length)
                    .sum::<usize>()
            );
            println!(
                "Chain code storage: {} bits (vs {} unpacked: {:.2}x compression)",
                total_packed_chain_bits,
                total_original_chain_bits,
                if total_packed_chain_bits > 0 {
                    total_original_chain_bits as f64 / total_packed_chain_bits as f64
                } else {
                    1.0
                }
            );
            println!(
                "Chain code overhead: {:.1} KB",
                total_packed_chain_bits as f64 / 8192.0
            );
        }
        Err(e) => {
            eprintln!("Error processing directory: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}
