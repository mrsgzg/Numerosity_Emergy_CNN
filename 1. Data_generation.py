import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from tqdm import tqdm
import random
import pandas as pd
from scipy.spatial import ConvexHull

class NumerosityStimuli:
    """Class for generating different types of numerosity stimulus datasets with improved control"""
    
    def __init__(self, image_size=(320, 240), background_color=0, 
                 dot_color=255, min_radius=3, max_radius=12,
                 min_distance=5, margin=20, seed=None):
        """
        Initialize the stimulus generator
        
        Args:
            image_size: Image size, (width, height)
            background_color: Background color (0-255)
            dot_color: Dot color (0-255)
            min_radius: Minimum dot radius
            max_radius: Maximum dot radius
            min_distance: Minimum distance between dots
            margin: Margin from image edge
            seed: Random seed
        """
        self.image_size = image_size
        self.background_color = background_color
        self.dot_color = dot_color
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_distance = min_distance
        self.margin = margin
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def _create_empty_image(self):
        """Create an empty image"""
        return np.ones(self.image_size[::-1], dtype=np.uint8) * self.background_color
    
    def _is_valid_position(self, x, y, radius, positions):
        """Check if a new position overlaps with existing dots"""
        if (x - radius < self.margin or 
            x + radius > self.image_size[0] - self.margin or
            y - radius < self.margin or 
            y + radius > self.image_size[1] - self.margin):
            return False
        
        for pos, r in positions:
            dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if dist < radius + r + self.min_distance:
                return False
        
        return True
    
    def _place_random_dots(self, num_dots, min_r=None, max_r=None):
        """Randomly place dots and return positions and radii"""
        if min_r is None:
            min_r = self.min_radius
        if max_r is None:
            max_r = self.max_radius
            
        positions = []
        for _ in range(num_dots):
            max_attempts = 1000
            attempt = 0
            placed = False
            
            while attempt < max_attempts and not placed:
                radius = np.random.uniform(min_r, max_r)
                x = np.random.uniform(self.margin + radius, self.image_size[0] - self.margin - radius)
                y = np.random.uniform(self.margin + radius, self.image_size[1] - self.margin - radius)
                
                if self._is_valid_position(x, y, radius, positions):
                    positions.append(((x, y), radius))
                    placed = True
                
                attempt += 1
                
            if not placed:
                print(f"Warning: Could not place all {num_dots} dots. Placed {len(positions)} dots.")
                break
                
        return positions
    
    def generate_standard_set(self, num_dots, num_images=10, random_contrast=False):
        """
        Generate standard stimulus set (circular dots of random size and spacing)
        
        Args:
            num_dots: Number of dots
            num_images: Number of images to generate
            random_contrast: If True, use random contrast for each dot
            
        Returns:
            List of generated images
        """
        images = []
        
        for _ in range(num_images):
            img = self._create_empty_image()
            positions = self._place_random_dots(num_dots)
            
            for (x, y), radius in positions:
                contrast = np.random.randint(100, 256) if random_contrast else self.dot_color
                cv2.circle(img, (int(x), int(y)), int(radius), contrast, -1)
            
            images.append(img)
        
        return images
    
    def generate_area_density_control(self, num_dots, num_images=10, total_area=1200, random_contrast=False):
        """
        Generate stimulus set with controlled total area and density
        
        Args:
            num_dots: Number of dots
            num_images: Number of images to generate
            total_area: Total area of all dots (pixels)
            random_contrast: If True, use random contrast for each dot
            
        Returns:
            List of generated images
        """
        images = []
        
        # Calculate area per dot, keeping total area constant
        area_per_dot = total_area / num_dots if num_dots > 0 else 0
        
        # Calculate corresponding radius
        radius = np.sqrt(area_per_dot / np.pi)
        
        for _ in range(num_images):
            img = self._create_empty_image()
            positions = self._place_random_dots(num_dots, min_r=radius, max_r=radius)
            
            for (x, y), r in positions:
                contrast = np.random.randint(100, 256) if random_contrast else self.dot_color
                cv2.circle(img, (int(x), int(y)), int(r), contrast, -1)
            
            images.append(img)
        
        return images
    
    def _generate_random_shape(self, center, size, color=None):
        """Generate random shape (circle, rectangle, ellipse or triangle) with optional color"""
        shape_type = np.random.choice(['circle', 'rectangle', 'ellipse', 'triangle'])
        x, y = center
        
        if color is None:
            color = self.dot_color
            
        if shape_type == 'circle':
            return lambda img: cv2.circle(img, (int(x), int(y)), int(size), color, -1)
        
        elif shape_type == 'rectangle':
            width = size * np.random.uniform(0.8, 1.2)
            height = size * np.random.uniform(0.8, 1.2)
            
            # Create rotated rectangle
            rect = ((x, y), (width*2, height*2), np.random.uniform(0, 180))
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            
            return lambda img: cv2.drawContours(img, [box], 0, color, -1)
        
        elif shape_type == 'ellipse':
            width = size * np.random.uniform(0.8, 1.2)
            height = size * np.random.uniform(0.8, 1.2)
            
            return lambda img: cv2.ellipse(img, (int(x), int(y)), 
                                          (int(width), int(height)), 
                                          np.random.uniform(0, 180), 0, 360, color, -1)
        
        else:  # triangle
            # Generate three vertices for the triangle
            angles = np.random.uniform(0, 2*np.pi, 3)
            r = size * np.random.uniform(0.8, 1.2, 3)
            
            pts = []
            for angle, radius in zip(angles, r):
                px = x + radius * np.cos(angle)
                py = y + radius * np.sin(angle)
                pts.append([int(px), int(py)])
            
            pts = np.array(pts, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            return lambda img: cv2.drawContours(img, [pts], 0, color, -1)
    
    def generate_shape_convex_hull_control(self, num_dots, num_images=10, random_contrast=False):
        """
        Generate stimulus set with controlled shape and convex hull
        
        Args:
            num_dots: Number of dots
            num_images: Number of images to generate
            random_contrast: If True, use random contrast for each shape
            
        Returns:
            List of generated images
        """
        images = []
        
        # Use similar convex hull for different numbers of dots
        # For counts > 4, use a fixed perimeter pentagon
        use_fixed_convex_hull = num_dots > 4
        
        for _ in range(num_images):
            img = self._create_empty_image()
            
            if use_fixed_convex_hull:
                # Create a pentagon convex hull
                center_x = self.image_size[0] // 2
                center_y = self.image_size[1] // 2
                
                # Pentagon with fixed perimeter
                radius = min(self.image_size[0], self.image_size[1]) // 3
                
                # Generate pentagon vertices
                pentagon_points = []
                for angle in np.linspace(0, 2*np.pi, 6)[:-1]:  # 5 points
                    px = center_x + radius * np.cos(angle)
                    py = center_y + radius * np.sin(angle)
                    pentagon_points.append((px, py))
                
                # Distribute points uniformly within the pentagon
                positions = []
                max_attempts = 5000
                attempts = 0
                
                while len(positions) < num_dots and attempts < max_attempts:
                    # Choose two random vertices and place a point at a random position between them
                    idx1, idx2 = np.random.choice(len(pentagon_points), 2, replace=False)
                    p1 = pentagon_points[idx1]
                    p2 = pentagon_points[idx2]
                    
                    alpha = np.random.uniform(0.2, 0.8)  # Avoid too close to the edge
                    x = p1[0] * alpha + p2[0] * (1 - alpha)
                    y = p1[1] * alpha + p2[1] * (1 - alpha)
                    
                    # Shift toward center slightly
                    x = 0.8 * x + 0.2 * center_x
                    y = 0.8 * y + 0.2 * center_y
                    
                    # Random radius, but keep small to avoid overlap
                    radius = np.random.uniform(self.min_radius, self.max_radius * 0.8)
                    
                    if self._is_valid_position(x, y, radius, positions):
                        positions.append(((x, y), radius))
                    
                    attempts += 1
                
                # Generate random shapes
                for (x, y), size in positions:
                    color = np.random.randint(100, 256) if random_contrast else self.dot_color
                    shape_func = self._generate_random_shape((x, y), size, color)
                    shape_func(img)
            
            else:
                # For small counts, simply place random shapes
                positions = self._place_random_dots(num_dots)
                
                for (x, y), size in positions:
                    color = np.random.randint(100, 256) if random_contrast else self.dot_color
                    shape_func = self._generate_random_shape((x, y), size, color)
                    shape_func(img)
            
            images.append(img)
        
        return images
    
    def _calculate_accurate_contour_length(self, shape_params):
        """
        Calculate the accurate contour length for a shape
        
        Args:
            shape_params: Dictionary containing shape type and dimensions
            
        Returns:
            Float: The contour length of the shape
        """
        if shape_params['type'] == 'circle':
            return 2 * np.pi * shape_params['radius']
        
        elif shape_params['type'] == 'rectangle':
            width = shape_params['width']
            height = shape_params['height']
            return 2 * (width + height)
        
        elif shape_params['type'] == 'ellipse':
            a = shape_params['width']
            b = shape_params['height']
            # More accurate ellipse perimeter approximation (Ramanujan's formula)
            h = ((a - b) ** 2) / ((a + b) ** 2)
            return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
        
        elif shape_params['type'] == 'triangle':
            # Calculate the perimeter of the triangle using the distances between vertices
            p1, p2, p3 = shape_params['vertices']
            
            # Calculate side lengths
            side1 = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            side2 = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
            side3 = np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
            
            return side1 + side2 + side3
        
        return 0
    
    def _draw_shape_with_params(self, img, shape_params, color):
        """
        Draw a shape on the image using the provided parameters
        
        Args:
            img: Image to draw on
            shape_params: Dictionary containing shape type and dimensions
            color: Shape color
            
        Returns:
            contour: The contour of the drawn shape (for validation)
        """
        if shape_params['type'] == 'circle':
            x, y = shape_params['center']
            radius = shape_params['radius']
            cv2.circle(img, (int(x), int(y)), int(radius), color, -1)
            
            # Create contour for validation
            contour = np.zeros_like(img)
            cv2.circle(contour, (int(x), int(y)), int(radius), 255, 1)
            
        elif shape_params['type'] == 'rectangle':
            x, y = shape_params['center']
            width = shape_params['width']
            height = shape_params['height']
            angle = shape_params['angle']
            
            rect = ((x, y), (width*2, height*2), angle)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            cv2.drawContours(img, [box], 0, color, -1)
            
            # Create contour for validation
            contour = np.zeros_like(img)
            cv2.drawContours(contour, [box], 0, 255, 1)
            
        elif shape_params['type'] == 'ellipse':
            x, y = shape_params['center']
            width = shape_params['width']
            height = shape_params['height']
            angle = shape_params['angle']
            
            cv2.ellipse(img, (int(x), int(y)), (int(width), int(height)), 
                      angle, 0, 360, color, -1)
            
            # Create contour for validation
            contour = np.zeros_like(img)
            cv2.ellipse(contour, (int(x), int(y)), (int(width), int(height)), 
                       angle, 0, 360, 255, 1)
            
        elif shape_params['type'] == 'triangle':
            vertices = shape_params['vertices']
            pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
            cv2.drawContours(img, [pts], 0, color, -1)
            
            # Create contour for validation
            contour = np.zeros_like(img)
            cv2.drawContours(contour, [pts], 0, 255, 1)
        
        else:
            contour = np.zeros_like(img)
            
        return contour
    
    def _measure_actual_contour_length(self, contour_img):
        """
        Measure the actual contour length from a contour image
        
        Args:
            contour_img: Binary image with the contour
            
        Returns:
            Float: The measured contour length
        """
        contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            return cv2.arcLength(contours[0], True)
        return 0
    
    def _get_shape_params_for_target_length(self, shape_type, target_length, center):
        """
        Calculate shape parameters to achieve a target contour length
        
        Args:
            shape_type: Type of shape ('circle', 'rectangle', 'ellipse', 'triangle')
            target_length: Target contour length
            center: Center position (x, y)
            
        Returns:
            Dictionary: Shape parameters
        """
        x, y = center
        
        if shape_type == 'circle':
            # 2πr = target_length => r = target_length / (2π)
            radius = target_length / (2 * np.pi)
            return {
                'type': 'circle',
                'center': (x, y),
                'radius': radius
            }
        
        elif shape_type == 'rectangle':
            # Assume width ≈ height ratio between 0.8 and 1.2
            ratio = np.random.uniform(0.8, 1.2)
            
            # For a rectangle with perimeter P and width:height ratio r
            # P = 2(w + h) = 2(w + w/r) = 2w(1 + 1/r)
            # Therefore, w = P / (2(1 + 1/r))
            width = target_length / (2 * (1 + 1/ratio))
            height = width / ratio
            
            return {
                'type': 'rectangle',
                'center': (x, y),
                'width': width,
                'height': height,
                'angle': np.random.uniform(0, 180)
            }
        
        elif shape_type == 'ellipse':
            # Use ratio between semi-major and semi-minor axes
            ratio = np.random.uniform(0.7, 1.3)
            
            # Approximate target using Ramanujan's formula
            # We'll use an iterative approach to find a and b
            # Start with a circle and adjust
            a = target_length / (2 * np.pi)  # Semi-major axis
            b = a / ratio  # Semi-minor axis
            
            # Iterative refinement to get closer to target perimeter
            for _ in range(3):  # Few iterations should be enough
                h = ((a - b) ** 2) / ((a + b) ** 2)
                current_perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
                
                # Scale both axes to approach target
                scale = target_length / current_perimeter
                a *= scale
                b *= scale
            
            return {
                'type': 'ellipse',
                'center': (x, y),
                'width': a,
                'height': b,
                'angle': np.random.uniform(0, 180)
            }
        
        else:  # triangle
            # For an approximately equilateral triangle
            # with perimeter P, side length s = P/3
            side_length = target_length / 3
            
            # Create slightly irregular triangle with controlled perimeter
            vertices = []
            base_radius = side_length / np.sqrt(3)  # Radius of circumscribed circle
            
            # Generate vertices with some controlled randomness
            for i in range(3):
                angle = 2 * np.pi * i / 3 + np.random.uniform(-0.2, 0.2)
                radius = base_radius * np.random.uniform(0.9, 1.1)
                px = x + radius * np.cos(angle)
                py = y + radius * np.sin(angle)
                vertices.append([int(px), int(py)])
            
            # Calculate actual perimeter
            p1, p2, p3 = vertices
            side1 = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            side2 = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
            side3 = np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
            actual_perimeter = side1 + side2 + side3
            
            # Scale vertices to match target perimeter
            scale = target_length / actual_perimeter
            vertices = [
                [int(x + (v[0] - x) * scale), int(y + (v[1] - y) * scale)]
                for v in vertices
            ]
            
            return {
                'type': 'triangle',
                'center': (x, y),
                'vertices': vertices
            }
    
    def generate_contour_length_control(self, num_dots, num_images=10, total_contour_length=800, 
                                    random_contrast=False, validation=True):
        """
        Generate stimulus set with accurately controlled total contour length
        
        Args:
            num_dots: Number of dots
            num_images: Number of images to generate
            total_contour_length: Total contour length to maintain across numerosities
            random_contrast: If True, use random contrast for each shape
            validation: If True, validate the contour length (slower but more accurate)
            
        Returns:
            List of generated images
        """
        images = []
        
        # Add scaling factor to prevent excessive size for small numbers
        # This creates a more reasonable progression of contour length per item
        if num_dots <= 1:
            # Apply a scaling factor that reduces the total contour length for very small numbers
            scaling_factor = 0.8  # Scales from 0.3 to 1.0 as num_dots goes from 1 to 4
            adjusted_total_length = total_contour_length * scaling_factor
        else:
            adjusted_total_length = total_contour_length
        
        for img_idx in range(num_images):
            img = self._create_empty_image()
            
            # Calculate target contour length per item with the adjusted total
            target_length_per_item = adjusted_total_length / num_dots if num_dots > 0 else 0
            
            # Define a maximum allowable contour length per item based on image dimensions
            # This ensures no single shape becomes too large
            #max_contour_length = min(self.image_size[0], self.image_size[1]) * 0.4
            #target_length_per_item = min(target_length_per_item, max_contour_length)
            
            # Place dots with random positions
            base_positions = self._place_random_dots(num_dots)
            
            # Track total contour length for this image
            actual_total_length = 0
            
            # Store shape parameters for possible refinement
            shape_params_list = []
            
            # Generate shapes with controlled contour length
            for (x, y), _ in base_positions:
                # Randomly choose shape type
                shape_type = np.random.choice(['circle', 'rectangle', 'ellipse', 'triangle'])
                
                # Calculate shape parameters for target length
                shape_params = self._get_shape_params_for_target_length(
                    shape_type, target_length_per_item, (x, y))
                
                # Additional size constraints based on shape type
                if shape_type == 'circle':
                    radius = shape_params['radius']
                    # Limit radius to a reasonable size relative to image
                    max_allowed_radius = min(self.image_size[0], self.image_size[1]) * 0.15
                    if radius > max_allowed_radius:
                        shape_params['radius'] = max_allowed_radius
                    
                    # Ensure shape is within bounds
                    if not self._is_valid_position(x, y, shape_params['radius'], []):
                        # If too large, reduce and recalculate
                        radius = min(
                            shape_params['radius'], 
                            self.max_radius * 2,  # Allow slightly larger than default max_radius
                            self.image_size[0] - self.margin - x,
                            x - self.margin,
                            self.image_size[1] - self.margin - y,
                            y - self.margin
                        )
                        shape_params['radius'] = max(radius, self.min_radius)
                
                # Apply similar size constraints to other shapes
                elif shape_type == 'rectangle' or shape_type == 'ellipse':
                    max_dimension = min(self.image_size[0], self.image_size[1]) * 0.15
                    if shape_params['width'] > max_dimension:
                        # Maintain aspect ratio
                        aspect = shape_params['height'] / shape_params['width']
                        shape_params['width'] = max_dimension
                        shape_params['height'] = max_dimension * aspect
                
                elif shape_type == 'triangle':
                    # Check if any vertex is too far from center
                    center_x, center_y = shape_params['center']
                    max_distance = min(self.image_size[0], self.image_size[1]) * 0.15
                    
                    # Scale down triangle if needed
                    vertices = shape_params['vertices']
                    max_vertex_distance = max(
                        np.sqrt((v[0] - center_x)**2 + (v[1] - center_y)**2) 
                        for v in vertices
                    )
                    
                    if max_vertex_distance > max_distance:
                        scale = max_distance / max_vertex_distance
                        shape_params['vertices'] = [
                            [int(center_x + (v[0] - center_x) * scale), 
                            int(center_y + (v[1] - center_y) * scale)]
                            for v in vertices
                        ]
                
                shape_params_list.append(shape_params)
                
                # Calculate theoretical contour length
                theoretical_length = self._calculate_accurate_contour_length(shape_params)
                actual_total_length += theoretical_length
            
            # Rest of the function remains the same
            # If the average differs significantly from target, adjust
            if num_dots > 0:
                scale_factor = adjusted_total_length / actual_total_length
                
                # Scale all shapes to better match target
                for shape_params in shape_params_list:
                    if shape_params['type'] == 'circle':
                        shape_params['radius'] *= scale_factor
                    elif shape_params['type'] == 'rectangle':
                        shape_params['width'] *= scale_factor
                        shape_params['height'] *= scale_factor
                    elif shape_params['type'] == 'ellipse':
                        shape_params['width'] *= scale_factor
                        shape_params['height'] *= scale_factor
                    elif shape_params['type'] == 'triangle':
                        # Scale vertices relative to center
                        center_x, center_y = shape_params['center']
                        vertices = []
                        for v in shape_params['vertices']:
                            scaled_x = center_x + (v[0] - center_x) * scale_factor
                            scaled_y = center_y + (v[1] - center_y) * scale_factor
                            vertices.append([int(scaled_x), int(scaled_y)])
                        shape_params['vertices'] = vertices
            
            # Draw shapes and validate if required
            if validation and num_dots > 0:
                # Draw shapes and measure actual contour lengths
                temp_img = self._create_empty_image()
                measured_lengths = []
                
                for shape_params in shape_params_list:
                    color = np.random.randint(100, 256) if random_contrast else self.dot_color
                    contour_img = self._draw_shape_with_params(temp_img, shape_params, color)
                    
                    # Measure actual contour length
                    length = self._measure_actual_contour_length(contour_img)
                    measured_lengths.append(length)
                
                # Calculate overall scale factor to match target
                total_measured = sum(measured_lengths)
                if total_measured > 0:
                    final_scale = adjusted_total_length / total_measured
                    
                    # Fine-tune shapes based on measured lengths
                    for i, shape_params in enumerate(shape_params_list):
                        if shape_params['type'] == 'circle':
                            shape_params['radius'] *= final_scale
                        elif shape_params['type'] == 'rectangle':
                            shape_params['width'] *= final_scale
                            shape_params['height'] *= final_scale
                        elif shape_params['type'] == 'ellipse':
                            shape_params['width'] *= final_scale
                            shape_params['height'] *= final_scale
                        elif shape_params['type'] == 'triangle':
                            # Scale vertices relative to center
                            center_x, center_y = shape_params['center']
                            vertices = []
                            for v in shape_params['vertices']:
                                scaled_x = center_x + (v[0] - center_x) * final_scale
                                scaled_y = center_y + (v[1] - center_y) * final_scale
                                vertices.append([int(scaled_x), int(scaled_y)])
                            shape_params['vertices'] = vertices
            
            # Draw final shapes
            for shape_params in shape_params_list:
                color = np.random.randint(100, 256) if random_contrast else self.dot_color
                self._draw_shape_with_params(img, shape_params, color)
            
            images.append(img)
            
            # Optional: Add progress indicator for longer generations
            if img_idx % 10 == 0 and img_idx > 0 and num_images > 20:
                print(f"Generated {img_idx}/{num_images} images for numerosity {num_dots}")
        
        return images
def create_dataset_splits(output_dir, labels_df, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Create train/val/test splits"""
    np.random.seed(seed)
    
    # Create balanced splits for each count and stimulus type
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Group by count and stimulus type
    for numerosity in labels_df['numerosity'].unique():
        for stim_type in labels_df['stimulus_type'].unique():
            # Get all indices for this group
            indices = labels_df[
                (labels_df['numerosity'] == numerosity) & 
                (labels_df['stimulus_type'] == stim_type)
            ].index.tolist()
            
            # Shuffle randomly
            np.random.shuffle(indices)
            
            # Calculate split sizes
            n_total = len(indices)
            n_test = max(1, int(test_ratio * n_total))
            n_val = max(1, int(val_ratio * n_total))
            n_train = n_total - n_test - n_val
            
            # Split the data
            test_indices.extend(indices[:n_test])
            val_indices.extend(indices[n_test:n_test+n_val])
            train_indices.extend(indices[n_test+n_val:])
    
    # Create splits
    train_df = labels_df.iloc[train_indices].copy()
    val_df = labels_df.iloc[val_indices].copy()
    test_df = labels_df.iloc[test_indices].copy()
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Print split information
    print(f"Dataset splits complete:")
    print(f"  Training set: {len(train_df)} images")
    print(f"  Validation set: {len(val_df)} images")
    print(f"  Test set: {len(test_df)} images")

def create_preview(dataset_dir, output_path, nums_to_show=[1, 4, 8, 16], 
                  stimulus_types=['standard', 'area_density', 'shape_convex', 'contour_length']):
    """Create dataset preview image"""
    plt.figure(figsize=(len(stimulus_types) * 4, len(nums_to_show) * 2))
    
    for i, num in enumerate(nums_to_show):
        for j, stim_type in enumerate(stimulus_types):
            plt.subplot(len(nums_to_show), len(stimulus_types), i*len(stimulus_types) + j + 1)
            
            # Find first image of this type and count
            pattern = f"{stim_type}_{num:02d}_"
            img_dir = os.path.join(dataset_dir, stim_type)
            try:
                matching_files = [f for f in os.listdir(img_dir) if pattern in f]
                
                if matching_files:
                    img_path = os.path.join(img_dir, matching_files[0])
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    plt.imshow(img, cmap='gray')
                    plt.title(f"{stim_type}, n={num}")
                else:
                    plt.text(0.5, 0.5, "Image not found", 
                             horizontalalignment='center',
                             verticalalignment='center')
            except Exception as e:
                plt.text(0.5, 0.5, f"Error: {str(e)}", 
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize=8)
            
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_images_with_labels(images, output_dir, prefix, number, stimulus_type, start_idx=0):
    """Save images to specified directory and return label information"""
    os.makedirs(output_dir, exist_ok=True)
    
    label_info = []
    
    for i, img in enumerate(images):
        filename = f"{prefix}_{number:02d}_{i+start_idx:03d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        
        # Save label information including image path, count label and stimulus type
        label_info.append({
            'image_path': filepath,
            'numerosity': number,  # Actual count label
            'label': number ,   # 0-based label (for model training)
            'stimulus_type': stimulus_type
        })
    
    return label_info

def main():
    parser = argparse.ArgumentParser(description='Generate numerosity stimulus datasets and create label files')
    parser.add_argument('--output_dir', type=str, default='numerosity_datasets_final2', 
                        help='Output directory')
    parser.add_argument('--min_number', type=int, default=0, 
                        help='Minimum dot count')
    parser.add_argument('--max_number', type=int, default=50, 
                        help='Maximum dot count')
    parser.add_argument('--images_per_number', type=int, default=100, 
                        help='Number of images per count')
    parser.add_argument('--img_width', type=int, default=320, 
                        help='Image width')
    parser.add_argument('--img_height', type=int, default=240, 
                        help='Image height')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--random_contrast', action='store_true', default=True,
                        help='Use random contrast for each shape')
    parser.add_argument('--total_area', type=int, default=1200,
                        help='Total area for area-density control')
    parser.add_argument('--total_contour_length', type=int, default=400,
                        help='Total contour length for contour length control')
    parser.add_argument('--validation', action='store_true', default=True,
                        help='Enable contour length validation')
    
    args = parser.parse_args()
    
    # Create output directories
    standard_dir = os.path.join(args.output_dir, 'standard')
    area_density_dir = os.path.join(args.output_dir, 'area_density')
    shape_convex_dir = os.path.join(args.output_dir, 'shape_convex')
    contour_length_dir = os.path.join(args.output_dir, 'contour_length')
    
    os.makedirs(standard_dir, exist_ok=True)
    os.makedirs(area_density_dir, exist_ok=True)
    os.makedirs(shape_convex_dir, exist_ok=True)
    os.makedirs(contour_length_dir, exist_ok=True)
    
    # Create stimulus generator
    stimuli = NumerosityStimuli(
        image_size=(args.img_width, args.img_height),
        seed=args.seed
    )
    
    # Store all image label information
    all_labels = []
    
    # Generate images for each count
    for num_dots in tqdm(range(args.min_number, args.max_number + 1), desc="Generating datasets"):
        # Standard set
        standard_images = stimuli.generate_standard_set(
            num_dots, num_images=args.images_per_number, random_contrast=args.random_contrast)
        standard_labels = save_images_with_labels(
            standard_images, standard_dir, 'standard', num_dots, 'standard')
        all_labels.extend(standard_labels)
        
        # Control set for total area and density
        area_density_images = stimuli.generate_area_density_control(
            num_dots, num_images=args.images_per_number, 
            total_area=args.total_area, random_contrast=args.random_contrast)
        area_density_labels = save_images_with_labels(
            area_density_images, area_density_dir, 'area_density', num_dots, 'area_density')
        all_labels.extend(area_density_labels)
        
        # Control set for shape and convex hull
        shape_convex_images = stimuli.generate_shape_convex_hull_control(
            num_dots, num_images=args.images_per_number, random_contrast=args.random_contrast)
        shape_convex_labels = save_images_with_labels(
            shape_convex_images, shape_convex_dir, 'shape_convex', num_dots, 'shape_convex')
        all_labels.extend(shape_convex_labels)
        
        # Control set for contour length with improved accuracy
        contour_length_images = stimuli.generate_contour_length_control(
            num_dots, num_images=args.images_per_number, 
            total_contour_length=args.total_contour_length, 
            random_contrast=args.random_contrast,
            validation=args.validation)
        contour_length_labels = save_images_with_labels(
            contour_length_images, contour_length_dir, 'contour_length', num_dots, 'contour_length')
        all_labels.extend(contour_length_labels)
    
    # Save label information to CSV file
    labels_df = pd.DataFrame(all_labels)
    labels_csv_path = os.path.join(args.output_dir, 'labels.csv')
    labels_df.to_csv(labels_csv_path, index=False)
    
    # Create train/val/test splits
    create_dataset_splits(args.output_dir, labels_df)
    
    print(f"All images saved to {args.output_dir} directory")
    print(f"Label information saved to {labels_csv_path}")
    
    # Create a sample preview
    preview_path = os.path.join(args.output_dir, 'preview.png')
    create_preview(args.output_dir, preview_path, 
                  stimulus_types=['standard', 'area_density', 'shape_convex', 'contour_length'])
    print(f"Preview image saved to {preview_path}")

if __name__ == "__main__":
    main()

