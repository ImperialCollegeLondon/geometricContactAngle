import numpy as np
import tifffile
from scipy.spatial import cKDTree
from scipy.ndimage import label, binary_dilation
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from tqdm import tqdm
import re

class ContactAngleInterpolator:
    def __init__(self, image_path, contact_angle_file, solid_label, fluid2_label, fluid1_label, primary_drainage=False):

        self.image = tifffile.imread(image_path)
        self.shape = self.image.shape
        self.contact_angle_file = contact_angle_file
        self.primary_drainage = primary_drainage

        self.SOLID = solid_label
        self.OIL = fluid2_label
        self.BRINE = fluid1_label

        self.measurements = None
        self.interpolated_angles = None
        self.brine_only_pores = None
        self.oil_objects = None
        self.oil_object_angles = None
        
        print(f"Loaded image with shape: {self.shape}")
        print(f"Unique phases: {np.unique(self.image)}")
        if primary_drainage:
            print("Primary drainage mode enabled - will assign 30° to fully brine-occupied pores")
    
    def identify_brine_only_pores(self):
        print("\nIdentifying fully brine-occupied pores...")
        pore_mask = (self.image == self.OIL) | (self.image == self.BRINE)
        labeled_pores, num_pores = label(pore_mask)
        print(f"Found {num_pores} distinct pores")
        self.brine_only_pores = np.zeros_like(self.image, dtype=bool)
        brine_only_count = 0
        for pore_id in range(1, num_pores + 1):
            pore_region = (labeled_pores == pore_id)
            oil_in_pore = np.any(self.image[pore_region] == self.OIL)
            if not oil_in_pore:
                self.brine_only_pores[pore_region] = True
                brine_only_count += 1
        
        brine_only_voxels = np.sum(self.brine_only_pores)
        total_pore_voxels = np.sum(pore_mask)
        print(f"Found {brine_only_count} fully brine-occupied pores")
        print(f"Brine-only pore voxels: {brine_only_voxels:,} ({100*brine_only_voxels/total_pore_voxels:.1f}% of pore space)")
        
        return self.brine_only_pores
    
    def segment_oil_objects(self):
        """Segment oil phase into connected objects"""
        print("\nSegmenting oil objects...")

        oil_mask = (self.image == self.OIL)

        self.oil_objects, num_objects = label(oil_mask)
        print(f"Found {num_objects} oil objects")

        object_sizes = []
        for obj_id in range(1, num_objects + 1):
            size = np.sum(self.oil_objects == obj_id)
            object_sizes.append((obj_id, size))
        
        object_sizes.sort(key=lambda x: x[1], reverse=True)
        if object_sizes:
            print(f"Largest oil object: {object_sizes[0][1]:,} voxels")
            print(f"Smallest oil object: {object_sizes[-1][1]:,} voxels")
        else:
            print("WARNING: No oil objects found!")
        
        return self.oil_objects
    
    def find_contact_lines_for_object(self, obj_id):

        obj_mask = (self.oil_objects == obj_id)

        dilated_obj = binary_dilation(obj_mask, iterations=3)
        nearby_lines = set()
        for idx, row in self.measurements.iterrows():
            voxel_pos = (int(row['voxel_z']), int(row['voxel_y']), int(row['voxel_x']))
            if (0 <= voxel_pos[0] < self.shape[0] and 
                0 <= voxel_pos[1] < self.shape[1] and 
                0 <= voxel_pos[2] < self.shape[2]):
                if dilated_obj[voxel_pos]:
                    nearby_lines.add(row['loop'])
        
        return list(nearby_lines)
    
    def calculate_line_uncertainty_weights(self):
        line_counts = self.measurements.groupby('loop').size()
        
        # Calculate weights (more points = higher weight = lower uncertainty)
        # Using sqrt to avoid over-weighting lines with many points
        weights = np.sqrt(line_counts)
        weights = weights / weights.max()  # Normalize to [0, 1]
        return weights
    
    def interpolate_oil_objects(self, method='weighted_mean', interpolation_threshold=3):
        
        if self.oil_objects is None:
            self.segment_oil_objects()
        
        if self.measurements is None:
            self.parse_contact_angle_file()
            self.map_coordinates_to_voxels()
        
        print("\nInterpolating contact angles for oil objects...")
        
        self.oil_object_angles = np.full(self.shape, np.nan)
        
        line_weights = self.calculate_line_uncertainty_weights()
        
        if method == 'weighted_mode':
            line_values = self.measurements.groupby('loop')['angle'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean())
        else:
            line_values = self.measurements.groupby('loop')['angle'].mean()
        
        num_objects = self.oil_objects.max()
        if num_objects == 0:
            print("WARNING: No oil objects found to interpolate!")
            return self.oil_object_angles
        
        objects_with_angles = 0
        
        for obj_id in tqdm(range(1, num_objects + 1), desc="Processing oil objects"):
            contact_lines = self.find_contact_lines_for_object(obj_id)
            
            if len(contact_lines) == 0:
                continue
            
            objects_with_angles += 1
            obj_mask = (self.oil_objects == obj_id)
            
            if len(contact_lines) <= interpolation_threshold:
                angles = []
                weights = []
                
                for line in contact_lines:
                    angles.append(line_values[line])
                    weights.append(line_weights[line])
                
                weighted_angle = np.average(angles, weights=weights)
                
                self.oil_object_angles[obj_mask] = weighted_angle
                
            else:
                relevant_measurements = self.measurements[self.measurements['loop'].isin(contact_lines)]
                
                points = relevant_measurements[['voxel_z', 'voxel_y', 'voxel_x']].values
                values = relevant_measurements['angle'].values
                
                point_weights = np.array([line_weights[line] for line in relevant_measurements['loop']])
                obj_coords = np.column_stack(np.where(obj_mask))
                tree = cKDTree(points)
                
                for coord in obj_coords:
                    k = min(10, len(points))
                    distances, indices = tree.query(coord, k=k)
                    
                    if distances[0] == 0:
                        angle = values[indices[0]]
                    else:
                        spatial_weights = 1.0 / (distances ** 2)
                        combined_weights = spatial_weights * point_weights[indices]
                        combined_weights /= combined_weights.sum()
                        angle = np.sum(combined_weights * values[indices])
                    
                    self.oil_object_angles[coord[0], coord[1], coord[2]] = angle
        
        valid_angles = self.oil_object_angles[~np.isnan(self.oil_object_angles)]
        
        print(f"Oil objects with nearby contact lines: {objects_with_angles}/{num_objects}")
        
        if len(valid_angles) > 0:
            print(f"Assigned angles to {len(valid_angles):,} oil voxels")
            print(f"Oil phase angle range: {np.min(valid_angles):.1f}° - {np.max(valid_angles):.1f}°")
            print(f"Mean oil phase angle: {np.mean(valid_angles):.1f}°")
        else:
            print("WARNING: No oil voxels were assigned angles!")
            print("This may be due to:")
            print("  - Contact lines being too far from oil objects")
            print("  - Coordinate mapping issues")
            print("  - Scale mismatch between image and measurements")
        
        return self.oil_object_angles
    
    def interpolate_remaining_brine(self, method='idw', max_distance=50):

        print("\nInterpolating remaining brine voxels near contact loops...")
        
        brine_mask = (self.image == self.BRINE)
        if self.primary_drainage and self.brine_only_pores is not None:
            remaining_brine = brine_mask & (~self.brine_only_pores)
        else:
            remaining_brine = brine_mask
        
        measurement_points = self.measurements[['voxel_z', 'voxel_y', 'voxel_x']].values
        angles = self.measurements['angle'].values
        tree = cKDTree(measurement_points)
        
        brine_coords = np.column_stack(np.where(remaining_brine))
        
        if self.interpolated_angles is None:
            self.interpolated_angles = np.full(self.shape, np.nan)
        
        interpolated_count = 0
        
        for coord in tqdm(brine_coords, desc="Interpolating brine voxels"):
            distances, indices = tree.query(coord, k=min(10, len(measurement_points)))
            
            if distances[0] <= max_distance:
                if method == 'idw':
                    weights = 1.0 / (distances ** 2)
                    weights /= weights.sum()
                    angle = np.sum(weights * angles[indices])
                elif method == 'nearest':
                    angle = angles[indices[0]]
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                self.interpolated_angles[coord[0], coord[1], coord[2]] = angle
                interpolated_count += 1
        
        print(f"Interpolated {interpolated_count:,} brine voxels within {max_distance} voxels of measurements")
        
        return self.interpolated_angles
    
    def create_wettability_colormap(self):
        import matplotlib.cm as cm
        
        if self.interpolated_angles is not None:
            valid_angles = self.interpolated_angles[~np.isnan(self.interpolated_angles)]
            if len(valid_angles) > 0:
                max_angle_in_data = np.max(valid_angles)
                min_angle_in_data = np.min(valid_angles)
            else:
                max_angle_in_data = 180
                min_angle_in_data = 1
        else:
            if self.measurements is not None and len(self.measurements) > 0:
                max_angle_in_data = self.measurements['angle'].max()
                min_angle_in_data = self.measurements['angle'].min()
            else:
                max_angle_in_data = 180
                min_angle_in_data = 1
        
        min_angle = max(1, min_angle_in_data - 10)
        max_angle = min(180, max_angle_in_data + 10)
        base_cmap = cm.get_cmap('jet')
        n_colors = int(max_angle - min_angle + 1)
        colors = []
        
        for i in range(n_colors):
            norm_pos = i / (n_colors - 1)
            color = base_cmap(norm_pos)
            colors.append(color[:3])

        cmap = ListedColormap(colors)
        cmap.set_bad(color='white')
        
        self.cmap_min = min_angle
        self.cmap_max = max_angle
        
        return cmap
    
    def parse_contact_angle_file(self, use_line_means=True):
        measurements = []
        
        with open(self.contact_angle_file, 'r') as f:
            lines = f.readlines()

        line_means = {}
        line_modes = {}
        line_counts = {}
        pl_section = False
        for ln in lines:
            if "PER-LOOP STATISTICS" in ln:
                pl_section = True
                continue
            if pl_section:
                if "INDIVIDUAL MEASUREMENTS" in ln or not ln.strip():
                    break
                # Match: Loop # | Points | Mean (°) | Mode (°) | Std Dev (°)
                m = re.match(r"\s*(\d+)\s+\|\s+(\d+)\s+\|\s+([\d\.]+)\s+\|\s+([\d\.]+)", ln)
                if m:
                    line_no = int(m.group(1))
                    count = int(m.group(2))
                    mean_ang = float(m.group(3))
                    mode_ang = float(m.group(4))
                    line_means[line_no] = mean_ang
                    line_modes[line_no] = mode_ang
                    line_counts[line_no] = count

        if not line_means:
            raise ValueError("Could not parse PER-LOOP STATISTICS section")

        start_idx = None
        for i, ln in enumerate(lines):
            if "INDIVIDUAL MEASUREMENTS" in ln:
                start_idx = i + 3
                break
        if start_idx is None:
            raise ValueError("Could not find INDIVIDUAL MEASUREMENTS section")

        records = []
        for ln in lines[start_idx:]:
            if not ln.strip():
                continue
            if "|" not in ln:
                continue
            left, angle_txt = ln.split("|")
            parts = left.split()
            if len(parts) < 5:
                continue
            try:
                line_no = int(parts[0])
                point_no = int(parts[1])
                x, y, z = map(float, parts[2:5])
                raw_angle = float(angle_txt)
            except ValueError:
                continue

            records.append(
                dict(loop=line_no, point=point_no,
                    x=x, y=y, z=z,
                    angle=line_modes[line_no] if use_line_means else raw_angle,
                    mean_angle=line_means[line_no],
                    mode_angle=line_modes[line_no],
                    raw_angle=raw_angle,
                    loop_count=line_counts[line_no])
            )

        self.measurements = pd.DataFrame(records)
        
        print(f"\nParsed {len(self.measurements)} points across "
            f"{self.measurements['loop'].nunique()} loops")
        print("Using line modes for angle values")
        
        line_summary = self.measurements.groupby('loop').agg({
            'mode_angle': 'first',
            'loop_count': 'first'
        })
        print("\nLoop statistics:")
        print(line_summary.head(10))  # Show first 10 loops

        return self.measurements
    
    def map_coordinates_to_voxels(self, scale_to_image=True):
        """Map measurement coordinates to voxel indices"""
        if self.measurements is None:
            self.parse_contact_angle_file()
        
        x_min, x_max = self.measurements['x'].min(), self.measurements['x'].max()
        y_min, y_max = self.measurements['y'].min(), self.measurements['y'].max()
        z_min, z_max = self.measurements['z'].min(), self.measurements['z'].max()
        
        print(f"\nMeasurement coordinate ranges:")
        print(f"X: [{x_min:.2f}, {x_max:.2f}]")
        print(f"Y: [{y_min:.2f}, {y_max:.2f}]")
        print(f"Z: [{z_min:.2f}, {z_max:.2f}]")
        
        if scale_to_image:
            self.measurements['voxel_z'] = ((self.measurements['z'] - z_min) / (z_max - z_min) * (self.shape[0] - 1)).round().astype(int)
            self.measurements['voxel_y'] = ((self.measurements['y'] - y_min) / (y_max - y_min) * (self.shape[1] - 1)).round().astype(int)
            self.measurements['voxel_x'] = ((self.measurements['x'] - x_min) / (x_max - x_min) * (self.shape[2] - 1)).round().astype(int)
            
            self.measurements['voxel_z'] = np.clip(self.measurements['voxel_z'], 0, self.shape[0] - 1)
            self.measurements['voxel_y'] = np.clip(self.measurements['voxel_y'], 0, self.shape[1] - 1)
            self.measurements['voxel_x'] = np.clip(self.measurements['voxel_x'], 0, self.shape[2] - 1)
        else:
            self.measurements['voxel_z'] = self.measurements['z'].round().astype(int)
            self.measurements['voxel_y'] = self.measurements['y'].round().astype(int)
            self.measurements['voxel_x'] = self.measurements['x'].round().astype(int)
        
        # Debug: Check if measurement voxels overlap with oil regions
        oil_voxel_count = 0
        for _, row in self.measurements.iterrows():
            z, y, x = int(row['voxel_z']), int(row['voxel_y']), int(row['voxel_x'])
            if (0 <= z < self.shape[0] and 0 <= y < self.shape[1] and 0 <= x < self.shape[2]):
                if self.image[z, y, x] == self.OIL:
                    oil_voxel_count += 1
        
        print(f"\nMeasurement points directly on oil voxels: {oil_voxel_count}/{len(self.measurements)}")
    
    def interpolate_angles(self, oil_method='weighted_mean', brine_method='idw', **kwargs):
        if self.measurements is None:
            self.parse_contact_angle_file()
            self.map_coordinates_to_voxels()
        
        self.interpolated_angles = np.full(self.shape, np.nan)
        
        if self.primary_drainage:
            self.identify_brine_only_pores()
            if self.brine_only_pores is not None:
                self.interpolated_angles[self.brine_only_pores] = 30
                print(f"Assigned 30° to {np.sum(self.brine_only_pores):,} voxels in brine-only pores")

        oil_angles = self.interpolate_oil_objects(
            method=oil_method,
            interpolation_threshold=kwargs.get('interpolation_threshold', 3)
        )

        oil_mask = ~np.isnan(oil_angles)
        self.interpolated_angles[oil_mask] = oil_angles[oil_mask]

        self.interpolate_remaining_brine(
            method=brine_method,
            max_distance=kwargs.get('max_distance', 20)
        )

        valid_mask = ~np.isnan(self.interpolated_angles)
        if np.any(valid_mask):
            self.interpolated_angles[valid_mask] = np.clip(self.interpolated_angles[valid_mask], 1, 180)
            valid_angles = self.interpolated_angles[~np.isnan(self.interpolated_angles)]
            print(f"\nFinal interpolation complete!")
            print(f"Total interpolated voxels: {len(valid_angles):,}")
            print(f"Angle range: {np.min(valid_angles):.1f}° - {np.max(valid_angles):.1f}°")
            print(f"Mean angle: {np.mean(valid_angles):.1f}°")
        else:
            print("\nWARNING: No voxels were interpolated!")
            print("Check that measurement coordinates align with the image dimensions.")
        
        return self.interpolated_angles
    
    def visualize_results(self, slice_indices=None, save_path=None):
        if self.interpolated_angles is None:
            raise ValueError("Run interpolation first!")
        
        valid_angles = self.interpolated_angles[~np.isnan(self.interpolated_angles)]
        if len(valid_angles) == 0:
            print("WARNING: No interpolated angles to visualize!")
            return
            
        wettability_cmap = self.create_wettability_colormap()
        fig = plt.figure(figsize=(11.69, 8.27))
        
        ax1 = plt.subplot(2, 2, 1)
        
        measurement_angles = np.full(self.shape, np.nan)
        for _, row in self.measurements.iterrows():
            z, y, x = int(row['voxel_z']), int(row['voxel_y']), int(row['voxel_x'])
            if (0 <= z < self.shape[0] and 0 <= y < self.shape[1] and 0 <= x < self.shape[2]):
                measurement_angles[z, y, x] = row['angle']
        
        raw_xy_projection = np.nanmean(measurement_angles, axis=0)
        
        masked_raw = np.ma.masked_where(np.isnan(raw_xy_projection), raw_xy_projection)
        vmin = 20
        vmax = 120
        im1 = ax1.imshow(masked_raw, cmap=wettability_cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax1.set_title('XY Projection - Raw Measurements', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X', fontsize=10)
        ax1.set_ylabel('Y', fontsize=10)
        cbar1 = plt.colorbar(im1, ax=ax1, label='Contact Angle (°)')
        
        cbar1.ax.axhline(y=70, color='black', linewidth=1, alpha=0.5)
        cbar1.ax.axhline(y=110, color='black', linewidth=1, alpha=0.5)
        
        ax2 = plt.subplot(2, 2, 2)

        interpolated_xy_projection = np.nanmean(self.interpolated_angles, axis=0)

        pore_mask_xy = np.any((self.image == self.OIL) | (self.image == self.BRINE), axis=0)
        masked_interp = np.ma.masked_where(~pore_mask_xy, interpolated_xy_projection)
        
        im2 = ax2.imshow(masked_interp, cmap=wettability_cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax2.set_title('XY Projection - Interpolated', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X', fontsize=10)
        ax2.set_ylabel('Y', fontsize=10)
        cbar2 = plt.colorbar(im2, ax=ax2, label='Contact Angle (°)')
        cbar2.ax.axhline(y=70, color='black', linewidth=1, alpha=0.5)
        cbar2.ax.axhline(y=110, color='black', linewidth=1, alpha=0.5)

        ax3 = plt.subplot(2, 2, 3)
        
        line_stats = self.measurements.groupby('loop').agg({
            'mean_angle': 'first',
            'loop_count': 'first',
            'mode_angle': 'first'
        })
        x = line_stats.index

        colors = []
        for angle in line_stats['mean_angle']:
            if angle < 70:
                colors.append('blue')
            elif angle <= 110:
                t = (angle - 70) / 40
                colors.append((t, 0, 1-t))
            else:
                colors.append('red')

        sizes = 30 + (line_stats['loop_count'] / line_stats['loop_count'].max()) * 150
        
        scatter = ax3.scatter(x, line_stats['mode_angle'], c=colors, s=sizes, 
                        edgecolor='black', alpha=0.7, linewidth=1)

        ax3.axhline(y=70, color='black', linestyle='--', alpha=0.3, label='Wettability boundaries')
        ax3.axhline(y=110, color='black', linestyle='--', alpha=0.3)
        ax3.text(ax3.get_xlim()[0] + 0.02*(ax3.get_xlim()[1]-ax3.get_xlim()[0]), 35, 
                'Water-wet', fontsize=8, style='italic', alpha=0.7)
        ax3.text(ax3.get_xlim()[0] + 0.02*(ax3.get_xlim()[1]-ax3.get_xlim()[0]), 90, 
                'Neutral', fontsize=8, style='italic', alpha=0.7)
        ax3.text(ax3.get_xlim()[0] + 0.02*(ax3.get_xlim()[1]-ax3.get_xlim()[0]), 145, 
                'Oil-wet', fontsize=8, style='italic', alpha=0.7)
        
        ax3.set_xlabel('Loop index', fontsize=11)
        ax3.set_ylabel('Mean Contact Angle (°)', fontsize=11)
        ax3.set_title('Contact-Loops Mean Angles', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 180)
        
        ax4 = plt.subplot(2, 2, 4)
        valid_angles_after = self.interpolated_angles[~np.isnan(self.interpolated_angles)]
        
        if len(valid_angles_after) > 0:
            bins = np.arange(0, 181, 5)
            n1, _, _ = ax4.hist(self.measurements['angle'], bins=bins, alpha=0.5, 
                            label=f'Raw measurements (n={len(self.measurements)})', 
                            color='seagreen', edgecolor='black', density=True, linewidth=0.5)
            n2, _, _ = ax4.hist(valid_angles_after, bins=bins, alpha=0.6, 
                            label=f'Interpolated (n={len(valid_angles_after):,})', 
                            color='tomato', edgecolor='black', density=True, linewidth=0.5)
            ax4.axvline(x=70, color='black', linestyle='--', alpha=0.5, linewidth=2)
            ax4.axvline(x=110, color='black', linestyle='--', alpha=0.5, linewidth=2)
            ax4.axvspan(0, 70, alpha=0.1, color='blue', label='Water-wet')
            ax4.axvspan(70, 110, alpha=0.1, color='purple', label='Neutral')
            ax4.axvspan(110, 180, alpha=0.1, color='red', label='Oil-wet')
            ax4.set_xlabel('Contact Angle (°)', fontsize=11)
            ax4.set_ylabel('Density', fontsize=11)
            ax4.set_title('Contact Angle Distribution: Before vs After', fontsize=14, fontweight='bold')
            ax4.set_xlim(0, 180)
            ax4.legend(loc='upper right', fontsize=8)
            ax4.grid(True, alpha=0.3)

            water_wet = np.sum((valid_angles_after >= 1) & (valid_angles_after < 70))
            neutral_wet = np.sum((valid_angles_after >= 70) & (valid_angles_after <= 110))
            oil_wet = np.sum((valid_angles_after > 110) & (valid_angles_after <= 180))
            total = len(valid_angles_after)

            print("═" * 45)
            print("WETTABILITY DISTRIBUTION")
            print("═" * 45)
            print(f"{'Water-wet (1-69°):':<25} {water_wet:>8,} ({100*water_wet/total:>5.1f}%)")
            print(f"{'Neutral (70-110°):':<25} {neutral_wet:>8,} ({100*neutral_wet/total:>5.1f}%)")
            print(f"{'Oil-wet (111-180°):':<25} {oil_wet:>8,} ({100*oil_wet/total:>5.1f}%)")
            print()
            print("═" * 45)
            print("OVERALL STATISTICS")
            print("═" * 45)
            print(f"{'Total pore voxels:':<25} {total:>8,}")
            print(f"{'Mean angle:':<25} {np.mean(valid_angles_after):>8.1f}°")
            print(f"{'Median angle:':<25} {np.median(valid_angles_after):>8.1f}°")
            print(f"{'Std deviation:':<25} {np.std(valid_angles_after):>8.1f}°")
            print(f"{'Range:':<25} {np.min(valid_angles_after):>5.1f}° - {np.max(valid_angles_after):.1f}°")
        else:
            ax4.text(0.5, 0.5, 'No interpolated data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
        
        if self.primary_drainage and self.brine_only_pores is not None:
            brine_only_count = np.sum(self.brine_only_pores)
            print()
            print("═" * 45)
            print("PRIMARY DRAINAGE INFO")
            print("═" * 45)
            print(f"{'Brine-only pore voxels:':<25} {brine_only_count:>8,}")
            print(f"{'Oil objects processed:':<25} {self.oil_objects.max():>8}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, output_path, save_metadata=False):
        """Save interpolated contact angle field"""
        if self.interpolated_angles is None:
            raise ValueError("Run interpolation first!")

        valid_angles = self.interpolated_angles[~np.isnan(self.interpolated_angles)]
        if len(valid_angles) == 0:
            print("WARNING: No valid angles to save!")
            return

        tifffile.imwrite(output_path, self.interpolated_angles.astype(np.float32))
        print(f"\nSaved contact angle distribution to: {output_path}")
        
        if save_metadata:
            import json

            water_wet_count = np.sum((valid_angles >= 1) & (valid_angles < 70))
            neutral_wet_count = np.sum((valid_angles >= 70) & (valid_angles <= 110))
            oil_wet_count = np.sum((valid_angles > 110) & (valid_angles <= 180))
            total_pore = len(valid_angles)
            
            metadata = {
                'image_shape': list(self.shape),
                'primary_drainage_mode': self.primary_drainage,
                'num_measurements': len(self.measurements),
                'num_loops': int(self.measurements['loop'].nunique()),
                'num_oil_objects': int(self.oil_objects.max()) if self.oil_objects is not None else 0,
                'measured_angle_mean': float(self.measurements['angle'].mean()),
                'measured_angle_std': float(self.measurements['angle'].std()),
                'measured_angle_min': float(self.measurements['angle'].min()),
                'measured_angle_max': float(self.measurements['angle'].max()),
                'interpolated_angle_mean': float(np.mean(valid_angles)) if len(valid_angles) > 0 else 0,
                'interpolated_angle_std': float(np.std(valid_angles)) if len(valid_angles) > 0 else 0,
                'interpolated_angle_min': float(np.min(valid_angles)) if len(valid_angles) > 0 else 0,
                'interpolated_angle_max': float(np.max(valid_angles)) if len(valid_angles) > 0 else 0,
                'num_pore_voxels': int(total_pore),
                'wettability_distribution': {
                    'water_wet_voxels': int(water_wet_count),
                    'water_wet_percentage': float(100 * water_wet_count / total_pore) if total_pore > 0 else 0,
                    'neutral_wet_voxels': int(neutral_wet_count),
                    'neutral_wet_percentage': float(100 * neutral_wet_count / total_pore) if total_pore > 0 else 0,
                    'oil_wet_voxels': int(oil_wet_count),
                    'oil_wet_percentage': float(100 * oil_wet_count / total_pore) if total_pore > 0 else 0
                }
            }
            
            if self.primary_drainage and self.brine_only_pores is not None:
                metadata['brine_only_voxels'] = int(np.sum(self.brine_only_pores))
                metadata['brine_only_percentage'] = float(100 * np.sum(self.brine_only_pores) / total_pore) if total_pore > 0 else 0

            metadata_path = output_path.replace('.tif', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to: {metadata_path}")
    
    def create_wettability_map(self):
        """Create ternary wettability map based on contact angle ranges"""
        if self.interpolated_angles is None:
            raise ValueError("Run interpolation first!")
        
        wettability_map = np.zeros_like(self.image)
        pore_mask = (self.image == self.OIL) | (self.image == self.BRINE)

        water_wet = pore_mask & (self.interpolated_angles >= 1) & (self.interpolated_angles < 70)
        wettability_map[water_wet] = 1

        neutral_wet = pore_mask & (self.interpolated_angles >= 70) & (self.interpolated_angles <= 110)
        wettability_map[neutral_wet] = 2

        oil_wet = pore_mask & (self.interpolated_angles > 110) & (self.interpolated_angles <= 180)
        wettability_map[oil_wet] = 3

        num_water_wet = np.sum(water_wet)
        num_neutral_wet = np.sum(neutral_wet)
        num_oil_wet = np.sum(oil_wet)
        total_pore = num_water_wet + num_neutral_wet + num_oil_wet
        
        if total_pore > 0:
            print(f"\nWettability Map Statistics:")
            print(f"Water-wet voxels (1-69°):   {num_water_wet:,} ({100*num_water_wet/total_pore:.1f}%)")
            print(f"Neutral-wet voxels (70-110°): {num_neutral_wet:,} ({100*num_neutral_wet/total_pore:.1f}%)")
            print(f"Oil-wet voxels (111-180°):   {num_oil_wet:,} ({100*num_oil_wet/total_pore:.1f}%)")
        else:
            print("\nWARNING: No pore voxels found in wettability map!")
            
        return wettability_map