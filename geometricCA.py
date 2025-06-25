import time
import os
from collections import defaultdict, Counter
from datetime import timedelta
import numpy as np
from Voxel2Surface import voxel_to_surfacev2
from SurfaceSmoothing import surface_smoothing
from NormalVectors import compute_contact_line_normals
from ContactLoops import identify_contact_loops

def compute_contact_angles2(contact_lines, ff_normals, fs_normals, filepath):
    all_angles = []
    all_records = []
    line_stats = {}
    
    # Extract filename without extension
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    
    print(f"Computing contact angles for {len(contact_lines)} contact loops...")
    
    for line_idx, (line, line_ff_normals, line_fs_normals) in enumerate(
            zip(contact_lines, ff_normals, fs_normals), start=1):
        try:
            pts = np.array(line)
            ffn = np.array(line_ff_normals)
            fsn = np.array(line_fs_normals)
            
            if not (len(pts) == len(ffn) == len(fsn)):
                print(f"Warning: Mismatched lengths in loop {line_idx}: "
                      f"Points={len(pts)}, FF_normals={len(ffn)}, FS_normals={len(fsn)}")
                continue
            
            ff_norms = np.linalg.norm(ffn, axis=1)
            fs_norms = np.linalg.norm(fsn, axis=1)
            
            valid_mask = (ff_norms > 1e-10) & (fs_norms > 1e-10)
            if not np.any(valid_mask):
                print(f"No valid normals found in loop {line_idx}")
                continue
            
            valid_pts = pts[valid_mask]
            valid_ffn = ffn[valid_mask]
            valid_fsn = fsn[valid_mask]
            valid_indices = np.nonzero(valid_mask)[0]
            
            valid_ffn = valid_ffn / ff_norms[valid_mask, np.newaxis]
            valid_fsn = valid_fsn / fs_norms[valid_mask, np.newaxis]
            
            dot_prods = np.einsum('ij,ij->i', valid_ffn, valid_fsn)
            angles = np.degrees(np.arccos(np.clip(dot_prods, -1.0, 1.0)))
            
            good_mask = ~np.isnan(angles)
            if not np.any(good_mask):
                print(f"No valid angle(s) after NaN filtering in line {line_idx}")
                continue
            
            final_pts = valid_pts[good_mask]
            final_angles = angles[good_mask]
            final_indices = valid_indices[good_mask]
            
            line_mean = float(np.mean(final_angles))
            line_mode = calculate_mode(final_angles)
            line_stats[line_idx] = {
                'mean': line_mean,
                'mode': line_mode,
                'count': len(final_angles),
                'std': float(np.std(final_angles))
            }
            
            for pt_idx, pt_coords, ang in zip(final_indices, final_pts, final_angles):
                all_records.append((line_idx, int(pt_idx), pt_coords.tolist(), float(ang)))
                all_angles.append(ang)
            
            print(f"Processed loop {line_idx}/{len(contact_lines)}: "
                  f"{len(final_angles)} valid angles, "
                  f"mean={line_mean:.2f}°, mode={line_mode:.2f}°")
        
        except Exception as e:
            print(f"Error processing loop {line_idx}: {e}")
            continue
    
    if not all_angles:
        print("No valid angles computed across all loops.")
        return [], np.nan, {}
    
    avg_angle = float(np.mean(all_angles))
    
    output_file = f"{base_name}_CAdata.txt"
    save_contact_angles2(avg_angle, all_records, line_stats, output_file)
    print(f"Overall average contact angle: {avg_angle:.2f}°")
    print(f"Total number of valid contact angles computed: {len(all_records)}")
    
    return all_records, avg_angle, line_stats

def calculate_mode(angles):
    if len(angles) == 0:
        return np.nan

    ints = np.rint(angles).astype(int)
    freq = Counter(ints)
    mode_int, _ = freq.most_common(1)[0]

    return float(mode_int)

def save_contact_angles2(avg_angle, contact_records, line_stats, filename):
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"OVERALL STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total contact loops analyzed: {len(line_stats)}\n")
        f.write(f"Total valid angle measurements: {len(contact_records)}\n")
        f.write(f"Overall average contact angle: {avg_angle:.2f}°\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("PER-LOOP STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write("Loop #  | Points | Mean (°) | Mode (°) | Std Dev (°)\n")
        f.write("-" * 60 + "\n")
        
        for line_idx in sorted(line_stats.keys()):
            stats = line_stats[line_idx]
            f.write(f"{line_idx:6d}  | {stats['count']:6d} | "
                   f"{stats['mean']:8.2f} | {stats['mode']:8.2f} | "
                   f"{stats['std']:10.2f}\n")
        
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("INDIVIDUAL MEASUREMENTS\n")
        f.write("=" * 80 + "\n")
        f.write("Loop  Point    x (units)   y (units)   z (units)   |  Angle (°)\n")
        f.write("-" * 60 + "\n")
        
        for line_idx, pt_idx, coords, angle in contact_records:
            x, y, z = coords
            f.write(
                f"{line_idx:4d}   "
                f"{pt_idx:5d}   "
                f"{x:8.4f}   {y:8.4f}   {z:8.4f}   |  {angle:5.2f}\n"
            )
    
    print(f"Saved contact angles to {filename}")
    print(f"  - Overall average: {avg_angle:.2f}°")
    print(f"  - Per-loop statistics for {len(line_stats)} loops")
    print(f"  - {len(contact_records)} individual measurements")

def compute_geometric_contact_angle(
    file_path,
    fluidLabel=1,
    solidLabel=2,
    minR_threshold=6,
    smoothing=True,
    min_points=20,
    smoothing_iter=50,
    edge_layers=1,
    normal_method="regression",
    max_euclidean_distance=15.0,
    use_polynomial_near_field=True,
    constraint_max_deviation=30.0,
    check_cross_consistency=True,
    enable_post_processing=True
):
    start = time.time()
    ff_mesh, fs_mesh = voxel_to_surfacev2(file_path, fluidLabel=fluidLabel, solidLabel=solidLabel, minR_threshold=minR_threshold)
    contact_lines = identify_contact_loops(ff_mesh, fs_mesh, smoothing=smoothing, min_points=min_points)
    ff_mesh, fs_mesh = surface_smoothing(ff_mesh, fs_mesh, iter=smoothing_iter, edge_layers=edge_layers)
    ff_normals, fs_normals, contact_points, contact_angles_raw = compute_contact_line_normals(
        ff_mesh, 
        fs_mesh, 
        contact_lines,
        method=normal_method,
        max_euclidean_distance=max_euclidean_distance,
        use_polynomial_near_field=use_polynomial_near_field,
        constraint_max_deviation=constraint_max_deviation,
        check_cross_consistency=check_cross_consistency,
        enable_post_processing=enable_post_processing
    )
    compute_contact_angles2(contact_points, ff_normals, fs_normals, file_path)
    elapsed = time.time() - start
    print(f"Measurement completed {str(timedelta(seconds=round(elapsed)))}")
