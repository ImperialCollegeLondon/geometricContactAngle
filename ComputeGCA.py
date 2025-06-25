import time
import os
from collections import defaultdict, Counter
from datetime import timedelta


import numpy as np
import statsmodels.api as sm
import tifffile
import trimesh
from scipy import ndimage
from scipy.interpolate import LSQUnivariateSpline
from scipy.spatial import cKDTree
from skimage import measure
from tqdm.auto import tqdm
from trimesh.smoothing import filter_taubin


def analyze_and_filter_phases(filename, fluidLabel, solidLabel, min_radius_threshold=6):
    
    volume = tifffile.imread(filename)
    volume_data = np.copy(volume).astype(np.float32)
    phase1_mask = (volume_data == fluidLabel)
    labeled_array, num_features = ndimage.label(phase1_mask)
    print(f"\nPhase 1 Analysis:")
    print(f"Initial number of clusters: {num_features}")
    props = measure.regionprops(labeled_array)
    volumes = np.array([prop.area for prop in props])
    radii = (3 * volumes / (4 * np.pi)) ** (1/3)
    component_ids = np.array([prop.label for prop in props])
    large_clusters = np.sum(radii >= min_radius_threshold)
    small_clusters = num_features - large_clusters
    print_statistics(radii, volumes, large_clusters, small_clusters, min_radius_threshold)
    keep_components = component_ids[radii >= min_radius_threshold]
    keep_mask = np.isin(labeled_array, keep_components)
    filtered_volume = volume_data.copy()
    filtered_volume[~keep_mask & (volume_data == fluidLabel)] = 0
    filtered_volume_data = np.copy(filtered_volume).astype(np.float32)
    filtered_volume_data[~np.isin(filtered_volume_data, [fluidLabel, solidLabel])] = 0
    return filtered_volume_data

def print_statistics(radii, volumes, large_clusters, small_clusters, min_radius_threshold):
    """Helper function to print analysis statistics"""
    print("\nCluster Statistics:")
    print(f"Total clusters detected: {len(radii)}")
    print(f"Clusters above threshold: {large_clusters}")
    print(f"Clusters below threshold: {small_clusters}")
    
    print(f"\nRadius Statistics:")
    print(f"Minimum radius: {np.min(radii):.2f} voxels")
    print(f"Maximum radius: {np.max(radii):.2f} voxels")
    print(f"Mean radius: {np.mean(radii):.2f} voxels")
    print(f"Median radius: {np.median(radii):.2f} voxels")
    print(f"Radius threshold: {min_radius_threshold:.2f} voxels")
    
    print("\nVolume Statistics:")
    print(f"Minimum volume: {np.min(volumes):.0f} voxels")
    print(f"Maximum volume: {np.max(volumes):.0f} voxels")
    print(f"Mean volume: {np.mean(volumes):.0f} voxels")
    print(f"Median volume: {np.median(volumes):.0f} voxels")

def extract_surface_mesh(volume_data, label):
    mask = (volume_data == label)
    verts, faces, normals, values = measure.marching_cubes(mask, gradient_direction='ascent')
    return verts, faces, normals

def find_common_vertices(verts1, verts2, tolerance=1e-6):
    tree = cKDTree(verts2)
    distances, indices = tree.query(verts1, distance_upper_bound=tolerance)
    common_mask = distances < tolerance
    
    return common_mask

def separate_interfaces(phase1_verts, phase1_faces, solid_verts, solid_faces):
    common_mask = find_common_vertices(phase1_verts, solid_verts)
    fluid_fluid_faces = []
    fluid_solid_faces = []
    for face in phase1_faces:
        if np.all(common_mask[face]):
            fluid_solid_faces.append(face)
        else:
            fluid_fluid_faces.append(face)
    fluid_fluid_faces = np.array(fluid_fluid_faces)
    fluid_solid_faces = np.array(fluid_solid_faces)
    ff_mesh = trimesh.Trimesh(vertices=phase1_verts, faces=fluid_fluid_faces)
    fs_mesh = trimesh.Trimesh(vertices=phase1_verts, faces=fluid_solid_faces)
    
    return ff_mesh, fs_mesh

def voxel_to_surfacev2(filename, fluidLabel=1, solidLabel=2, minR_threshold=4, save_mesh_files=False):
    print("Starting interface extraction process...")
    volume_data = analyze_and_filter_phases(filename, fluidLabel, solidLabel, minR_threshold)
    print(f"Image data loaded and filtered from {filename}")
    phase1_verts, phase1_faces, _ = extract_surface_mesh(volume_data, fluidLabel)
    solid_verts, solid_faces, _ = extract_surface_mesh(volume_data, solidLabel)
    print("Surface meshes extracted successfully")
    ff_mesh, fs_mesh = separate_interfaces(phase1_verts, phase1_faces, solid_verts, solid_faces)
    print(f"FF_interface Faces: {len(ff_mesh.faces)}, Vertices: {len(ff_mesh.vertices)}")
    print(f"FS_interface Faces: {len(fs_mesh.faces)}, Vertices: {len(fs_mesh.vertices)}")
    if len(ff_mesh.faces) == 0 or len(fs_mesh.faces) == 0:
        print("Error: One or both interfaces have no faces. Cannot proceed with contact angle calculation.")
        return
    print("FF_interface Boundaries:", ff_mesh.bounds)
    print("FS_interface Boundaries:", fs_mesh.bounds)
    print("Surface extraction process completed successfully.")
    return ff_mesh, fs_mesh

def identify_contact_loops(ff_mesh, fs_mesh, min_points=40, smoothing=True, loop_closure_threshold=5):
    print("Initializing contact loop identification...")
    boundary_edges = ff_mesh.edges_unique[ff_mesh.edges_unique_length > 0]
    boundary_vertices = np.unique(boundary_edges.flatten())
    boundary_points = ff_mesh.vertices[boundary_vertices]
    tree = cKDTree(fs_mesh.vertices)
    distances, _ = tree.query(boundary_points, distance_upper_bound=1e-6)
    common_mask = distances < 1e-6
    valid_vertices_mask = common_mask[boundary_vertices]
    vertex_index_map = {v: i for i, v in enumerate(boundary_vertices)}
    valid_edges_mask = valid_vertices_mask[boundary_edges[:, 0]] & valid_vertices_mask[boundary_edges[:, 1]]
    valid_edges = boundary_edges[valid_edges_mask]
    point_connections = defaultdict(set)
    for edge in valid_edges:
        point1 = tuple(ff_mesh.vertices[edge[0]])
        point2 = tuple(ff_mesh.vertices[edge[1]])
        point_connections[point1].add(point2)
        point_connections[point2].add(point1)
    contact_lines = []
    visited_points = set()
    for start_point in point_connections:
        if start_point not in visited_points:
            line = trace_contact_line(start_point, point_connections, visited_points, 
                                    loop_closure_threshold)
            if len(line) > 1:
                contact_lines.append(line)
    original_count = len(contact_lines)
    filtered_lines = [line for line in contact_lines if len(line) >= min_points]
    filtered_count = len(filtered_lines)
    loop_count = sum(1 for line in filtered_lines if line[0] == line[-1])
    print(f"\nContact Loop Analysis:")
    print(f"Original number of loops: {original_count}")
    print(f"Loops after length filtering (min {min_points} points): {filtered_count}")
    print(f"Number of closed loops: {loop_count}")
    print(f"Number of open loopss: {filtered_count - loop_count}")
    print(f"Removed {original_count - filtered_count} short loops")
    if not smoothing:
        return filtered_lines
    print("\nApplying B-spline smoothing...")
    smoothed_lines = process_contact_lines_bspline(filtered_lines)
    return smoothed_lines

def trace_contact_line(start_point, connections, visited, loop_closure_threshold):
    contact_line = [start_point]
    current_point = start_point
    while True:
        visited.add(current_point)
        connected_points = connections[current_point]
        next_point = None
        for point in connected_points:
            if point not in visited:
                next_point = point
                break
        if next_point is None:
            if len(contact_line) > 2:
                start_array = np.array(contact_line[0])
                end_array = np.array(contact_line[-1])
                distance = np.linalg.norm(end_array - start_array)
                if distance <= loop_closure_threshold:
                    contact_line.append(contact_line[0])
            break
        contact_line.append(next_point)
        current_point = next_point
    return contact_line

def process_contact_lines_bspline(contact_lines, degree=3, smoothing=0.1):
    smoothed_lines = []
    for i, line in enumerate(contact_lines):
        points = np.array(line)
        if len(points) < degree + 2:
            smoothed_lines.append(line)
            continue
        is_loop = np.array_equal(points[0], points[-1])
        if is_loop:
            n_repeat = degree + 1
            points_to_fit = np.vstack((points[:-1], points[1:n_repeat]))
        else:
            points_to_fit = points
        t = np.zeros(len(points_to_fit))
        for j in range(1, len(points_to_fit)):
            t[j] = t[j-1] + np.linalg.norm(points_to_fit[j] - points_to_fit[j-1])
        t = t / t[-1]
        n_knots = max(4, len(points) // 4)
        knots = np.linspace(0, 1, n_knots + 2)[1:-1]
        smoothed_coords = []
        for dim in range(3):
            try:
                spl = LSQUnivariateSpline(
                    t, points_to_fit[:, dim],
                    knots,
                    k=degree,
                    w=np.ones_like(t) * (1 - smoothing)
                )
                
                t_dense = np.linspace(0, 1 if not is_loop else t[-(n_repeat+1)], 
                                    len(points) * 2)
                smoothed_coords.append(spl(t_dense))
            except Exception as e:
                print(f"Error fitting dimension {dim} for loop {i+1}: {str(e)}")
                smoothed_lines.append(line)
                continue
        smoothed_points = np.column_stack(smoothed_coords)
        if is_loop:
            smoothed_points = np.vstack((smoothed_points, smoothed_points[0]))
            points_filtered = remove_redundant_points(smoothed_points[:-1])
            smoothed_points = np.vstack((points_filtered, points_filtered[0]))
        else:
            smoothed_points = remove_redundant_points(smoothed_points)
        
        smoothed_line = [tuple(point) for point in smoothed_points]
        smoothed_lines.append(smoothed_line)
        print(f"\nLoop {i+1} Statistics:")
        print(f"  Type: {'Closed Loop' if is_loop else 'Open Curve'}")
        print(f"  Points: {len(line)} --> {len(smoothed_line)}")
    return smoothed_lines

def remove_redundant_points(points, min_distance=0.8):
    if len(points) < 2:
        return points
    filtered_points = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - filtered_points[-1]) >= min_distance:
            filtered_points.append(point)
    return np.array(filtered_points)

def surface_smoothing(
    ff_mesh: trimesh.Trimesh,
    fs_mesh: trimesh.Trimesh,
    *,
    edge_layers: int = 2,
    lamb: float = 0.5,
    nu: float = 0.5,
    iter: int = 50,
):
    ff_v, ff_f = _remove_edge_layers(ff_mesh.vertices, ff_mesh.faces, edge_layers)
    ff_clean = trimesh.Trimesh(vertices=ff_v, faces=ff_f, process=False)
    ff_smooth = ff_clean.copy()
    filter_taubin(ff_smooth, lamb=lamb, nu=nu, iterations=iter*2)
    fs_smooth = fs_mesh.copy()
    filter_taubin(fs_smooth, lamb=lamb, nu=nu, iterations=iter*2)
    return ff_smooth, fs_smooth

def find_edge_faces(faces):
    edge_to_face = {}
    for f_idx, f in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((f[i], f[(i + 1) % 3])))
            edge_to_face.setdefault(e, []).append(f_idx)
    return [fi[0] for fi in edge_to_face.values() if len(fi) == 1]

def _remove_edge_layers(verts: np.ndarray, faces: np.ndarray, n_layers: int):
    cur_v, cur_f = verts.copy(), faces.copy()
    for lyr in range(n_layers):
        bnd = find_edge_faces(cur_f)
        if not bnd:
            print(f"No boundary faces left after stripping {lyr} layer(s).")
            break
        mask = np.ones(len(cur_f), bool)
        mask[bnd] = False
        cur_f = cur_f[mask]
        used = np.unique(cur_f)
        vmap = np.full(len(cur_v), -1, int)
        vmap[used] = np.arange(len(used))
        cur_v = cur_v[used]
        cur_f = vmap[cur_f]
    return cur_v, cur_f

def calculate_secant_direction(contact_line, current_index, window=2):
    n = len(contact_line)
    i0 = max(0, current_index - window)
    i1 = min(n - 1, current_index + window)
    sec = np.asarray(contact_line[i1]) - np.asarray(contact_line[i0])
    nrm = np.linalg.norm(sec)
    return sec / nrm if nrm > 1e-10 else np.array([1., 0., 0.])

def lowess_prediction(x, y, x_pred, frac=0.5):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    if not np.isclose(x, x_pred).any():
        x_vals = np.concatenate([x, [x_pred]])
    else:
        x_vals = x
    
    y_vals = sm.nonparametric.lowess(
        y, x, xvals=x_vals, frac=frac, is_sorted=False
    )

    idx = -1 if len(x_vals) > len(x) else np.argwhere(np.isclose(x_vals, x_pred))[0, 0]
    return y_vals[idx]

def _orient_normals(normals, ref_normal):
    mask = (normals @ ref_normal) < 0.0
    normals[mask] *= -1.0
    return normals

def polynomial_extrapolation_near_origin(distances, values, max_dist=10.0, degree=2):
    mask = distances <= max_dist
    if np.sum(mask) < degree + 1:
        return None
    
    x_near = distances[mask]
    y_near = values[mask]
    
    weights = np.exp(-x_near / (max_dist / 3))
    
    try:
        coeffs = np.polyfit(x_near, y_near, degree, w=weights)
        y_pred = np.polyval(coeffs, 0)
        
        y_mean = np.mean(y_near)
        if abs(y_pred - y_mean) > 2 * np.std(y_near):
            return np.sum(weights * y_near) / np.sum(weights)
        
        return y_pred
    except:
        return None

def constrained_normal_extrapolation(predicted_normal, reference_normals, max_deviation=30.0):
    if len(reference_normals) == 0:
        return predicted_normal
    
    pred_norm = predicted_normal / (np.linalg.norm(predicted_normal) + 1e-10)
    
    ref_mean = np.mean(reference_normals, axis=0)
    ref_mean_norm = ref_mean / (np.linalg.norm(ref_mean) + 1e-10)
    
    cos_angle = np.dot(pred_norm, ref_mean_norm)
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    if angle_deg > max_deviation:
        t = max_deviation / angle_deg
        
        omega = np.arccos(np.clip(cos_angle, -1, 1))
        if abs(omega) > 1e-6:
            constrained = (np.sin((1-t)*omega)/np.sin(omega)) * ref_mean_norm + \
                         (np.sin(t*omega)/np.sin(omega)) * pred_norm
        else:
            constrained = pred_norm
        
        return constrained
    
    return pred_norm

def check_cross_interface_consistency(ff_normals, fs_normals, ff_points, fs_points, 
                                     contact_point, radius=5.0):
    ff_dists = np.linalg.norm(ff_points - contact_point, axis=1)
    fs_dists = np.linalg.norm(fs_points - contact_point, axis=1)
    
    ff_near = ff_dists < radius
    fs_near = fs_dists < radius
    
    if np.sum(ff_near) < 3 or np.sum(fs_near) < 3:
        return 1.0
    
    angles = []
    for ff_n in ff_normals[ff_near]:
        for fs_n in fs_normals[fs_near]:
            cos_angle = np.dot(ff_n, fs_n)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles.append(angle)
    
    avg_angle = np.mean(angles)
    quality = min(avg_angle / 90.0, 1.0)
    return quality

def process_point(
    point, face_tree, face_normals, face_centers,
    contact_line, point_index,
    *,
    mesh_tag,
    method="regression",
    d=0.5, window=2, max_neighbors=50, min_points=20,
    orient_neighbors=True,
    max_euclidean_distance=15.0,
    lowess_improvements=True,
    use_polynomial_near_field=True,
    constraint_max_deviation=30.0,
):
    point = np.asarray(point, float)
    secant_dir = calculate_secant_direction(contact_line, point_index, window)

    tmp = np.array([0., 0., 1.]) if abs(secant_dir[2]) < .9 else np.array([1., 0., 0.])
    b2 = np.cross(secant_dir, tmp)
    b2 /= np.linalg.norm(b2)
    b3 = np.cross(secant_dir, b2)
    T = np.vstack([secant_dir, b2, b3])

    local = (face_centers - point) @ T.T
    slab = np.abs(local[:, 0]) <= d
    slice_face_idx = np.where(slab)[0]
    slice_pts = face_centers[slab]
    slice_normals = face_normals[slab]
    slice_locals = local[slab]

    if len(slice_pts) < 3:
        _, idx = face_tree.query(point, k=1)
        return face_normals[idx], slice_pts, None, slice_normals, 0.0

    euclid = np.linalg.norm(slice_pts - point, axis=1)
    
    effective_max_distance = max_euclidean_distance
    if mesh_tag == 'FF':
        effective_max_distance = min(max_euclidean_distance, 10.0)
    
    distance_mask = euclid <= effective_max_distance
    slice_face_idx = slice_face_idx[distance_mask]
    slice_pts = slice_pts[distance_mask]
    slice_normals = slice_normals[distance_mask]
    slice_locals = slice_locals[distance_mask]
    euclid = euclid[distance_mask]
    
    if len(slice_pts) < 3:
        _, idx = face_tree.query(point, k=1)
        return face_normals[idx], slice_pts, None, slice_normals, 0.0

    right_m, left_m = slice_locals[:, 1] >= 0, slice_locals[:, 1] <= 0
    r_pts, l_pts = slice_locals[right_m], slice_locals[left_m]
    r_spread = np.ptp(r_pts[:, 0]) if r_pts.size else 0
    l_spread = np.ptp(l_pts[:, 0]) if l_pts.size else 0
    use_whole = (
        len(r_pts) < min_points or
        len(l_pts) < min_points or
        min(r_spread, l_spread) < 0.3 * max(r_spread, l_spread)
    )
    if not use_whole:
        d1 = euclid
        nn5 = np.argsort(d1)[:5]
        hp = right_m if (slice_locals[nn5, 1] > 0).sum() >= len(nn5) / 2 else left_m
        slice_face_idx = slice_face_idx[hp]
        slice_pts, slice_normals, slice_locals, euclid = (
            slice_pts[hp], slice_normals[hp], slice_locals[hp], euclid[hp]
        )

    if len(slice_pts) > max_neighbors:
        keep = np.argsort(euclid)[:max_neighbors]
        slice_face_idx = slice_face_idx[keep]
        slice_pts, slice_normals, slice_locals, euclid = (
            slice_pts[keep], slice_normals[keep], slice_locals[keep], euclid[keep]
        )

    ref_normal = slice_normals[np.argmin(euclid)]
    if orient_neighbors:
        _orient_normals(slice_normals, ref_normal)

    method = method.lower()
    predicted_normal = np.zeros(3)

    if method == "weighted":
        w = np.exp(-euclid / 2.0)
        w /= w.sum()
        predicted_normal = (w[:, None] * slice_normals).sum(0)

    elif method == "regression":
        predicted_normal = np.zeros(3)

        if lowess_improvements:
            n_points = len(slice_normals)
            distance_spread = np.ptp(euclid)
            
            base_frac = min(0.7, max(0.2, 30. / n_points))
            
            if distance_spread > 0:
                density = n_points / distance_spread
                if density > 5:
                    frac = base_frac * 0.8
                elif density < 2:
                    frac = min(0.8, base_frac * 1.5)
                else:
                    frac = base_frac
            else:
                frac = base_frac
        else:
            frac = min(0.5, max(0.3, 20. / len(slice_normals)))

        order = np.argsort(euclid)
        x_sorted = euclid[order]

        for k in range(3):
            y_sorted = slice_normals[order, k]
            
            if use_polynomial_near_field:
                poly_pred = polynomial_extrapolation_near_origin(
                    x_sorted, y_sorted, max_dist=10, degree=2
                )
                
                if poly_pred is not None:
                    predicted_normal[k] = poly_pred
                    continue
            
            if lowess_improvements and len(y_sorted) > 10:
                y_fitted_initial = sm.nonparametric.lowess(
                    y_sorted, x_sorted, xvals=x_sorted, frac=frac, is_sorted=True
                )
                
                residuals = np.abs(y_sorted - y_fitted_initial)
                threshold = 2.5 * np.std(residuals)
                
                outlier_mask = residuals > threshold
                n_outliers = np.sum(outlier_mask)
                
                if 0 < n_outliers < 0.2 * len(y_sorted):
                    x_filtered = x_sorted[~outlier_mask]
                    y_filtered = y_sorted[~outlier_mask]
                    
                    if len(x_filtered) >= 3:
                        y_pred = lowess_prediction(
                            x_filtered, y_filtered, 0.0, frac=frac
                        )
                    else:
                        y_pred = lowess_prediction(
                            x_sorted, y_sorted, 0.0, frac=frac
                        )
                else:
                    y_pred = lowess_prediction(
                        x_sorted, y_sorted, 0.0, frac=frac
                    )
            else:
                y_pred = lowess_prediction(
                    x_sorted, y_sorted, 0.0, frac=frac
                )

            predicted_normal[k] = y_pred
    else:
        raise ValueError("method must be 'lowess' or 'weighted'.")

    nrm = np.linalg.norm(predicted_normal)
    if nrm > 1e-8:
        predicted_normal = predicted_normal / nrm
    else:
        predicted_normal = ref_normal / np.linalg.norm(ref_normal)
    
    if lowess_improvements:
        nearest_k = min(5, len(euclid))
        nearest_idx = np.argsort(euclid)[:nearest_k]
        reference_normals = slice_normals[nearest_idx]
        
        predicted_normal = constrained_normal_extrapolation(
            predicted_normal, reference_normals, 
            max_deviation=constraint_max_deviation
        )

    return predicted_normal, slice_pts, euclid, slice_normals, 0.0

def post_process_contact_angles(ff_normals, fs_normals, points, contact_angles,
                               smoothing_window=5, outlier_threshold=3.0):
    processed_angles = contact_angles.copy()
    ff_normals_proc = ff_normals.copy()
    fs_normals_proc = fs_normals.copy()
    n_points = len(contact_angles)
    
    for i in range(n_points):
        start = max(0, i - smoothing_window // 2)
        end = min(n_points, i + smoothing_window // 2 + 1)
        local_angles = contact_angles[start:end]
        
        if len(local_angles) > 3:
            median = np.median(local_angles)
            mad = np.median(np.abs(local_angles - median))
            
            if mad > 0:
                z_score = abs(contact_angles[i] - median) / (1.4826 * mad)
                
                if z_score > outlier_threshold and contact_angles[i] < 30:
                    ff_n = ff_normals[i]
                    fs_n = fs_normals[i]
                    
                    if np.dot(ff_n, fs_n) > 0.9:
                        target_angle = np.radians(median)
                        
                        axis = np.cross(ff_n, fs_n)
                        if np.linalg.norm(axis) > 1e-6:
                            axis /= np.linalg.norm(axis)
                            rotation_angle = target_angle - np.arccos(np.clip(np.dot(ff_n, fs_n), -1, 1))
                            c = np.cos(rotation_angle)
                            s = np.sin(rotation_angle)
                            rotation_matrix = (
                                c * np.eye(3) + 
                                s * np.array([[0, -axis[2], axis[1]],
                                             [axis[2], 0, -axis[0]],
                                             [-axis[1], axis[0], 0]]) +
                                (1 - c) * np.outer(axis, axis)
                            )
                            ff_normals_proc[i] = rotation_matrix @ ff_n
                            
                            processed_angles[i] = np.degrees(
                                np.arccos(np.clip(np.dot(ff_normals_proc[i], fs_n), -1, 1))
                            )
    
    return processed_angles, ff_normals_proc, fs_normals_proc

def compute_contact_line_normals(
    ff_mesh, fs_mesh, contact_lines,
    *,
    method: str = "regression",
    max_euclidean_distance: float = 15.0,
    lowess_improvements: bool = True,
    use_polynomial_near_field: bool = True,
    constraint_max_deviation: float = 30.0,
    enable_post_processing: bool = True,
    check_cross_consistency: bool = True,
):
    ff_normals, fs_normals = ff_mesh.face_normals, fs_mesh.face_normals
    ff_centers, fs_centers = ff_mesh.triangles_center, fs_mesh.triangles_center
    ff_tree, fs_tree = cKDTree(ff_centers), cKDTree(fs_centers)
    d = 0.5
    ff_out, fs_out, pts_out, angles_out = [], [], [], []

    for li, line in enumerate(contact_lines):
        ff_line_norms, fs_line_norms, pts_line, angles_line = [], [], [], []
        
        all_ff_results = []
        all_fs_results = []

        for pi, p in enumerate(
            tqdm(line, desc=f"Loop {li+1}/{len(contact_lines)}",
                 position=1, leave=False, dynamic_ncols=True)
        ):
            ff_result = process_point(
                p, ff_tree, ff_normals, ff_centers,
                line, pi, mesh_tag="FF",
                method=method, d=d,
                max_euclidean_distance=max_euclidean_distance,
                lowess_improvements=lowess_improvements,
                use_polynomial_near_field=use_polynomial_near_field,
                constraint_max_deviation=constraint_max_deviation
            )
            fs_result = process_point(
                p, fs_tree, fs_normals, fs_centers,
                line, pi, mesh_tag="FS",
                method=method, d=d,
                max_euclidean_distance=max_euclidean_distance,
                lowess_improvements=lowess_improvements,
                use_polynomial_near_field=use_polynomial_near_field,
                constraint_max_deviation=constraint_max_deviation
            )
            
            all_ff_results.append(ff_result)
            all_fs_results.append(fs_result)
            
            ff_n = ff_result[0]
            fs_n = fs_result[0]
            
            consistency_quality = 1.0
            if check_cross_consistency and ff_result[1] is not None and fs_result[1] is not None:
                consistency_quality = check_cross_interface_consistency(
                    ff_result[3], fs_result[3],
                    ff_result[1], fs_result[1],
                    p, radius=5.0
                )

            dot_prod = np.dot(ff_n, fs_n)
            contact_angle = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))
            
            skip_point = False
            
            if consistency_quality < 0.3:
                skip_point = True
            
            if not skip_point:
                ff_line_norms.append(ff_n)
                fs_line_norms.append(fs_n)
                pts_line.append(p)
                angles_line.append(contact_angle)

        if enable_post_processing and len(angles_line) > 0:
            angles_array = np.array(angles_line)
            ff_array = np.array(ff_line_norms)
            fs_array = np.array(fs_line_norms)
            pts_array = np.array(pts_line)
            
            proc_angles, ff_proc, fs_proc = post_process_contact_angles(
                ff_array, fs_array, pts_array, angles_array,
                smoothing_window=7,
                outlier_threshold=2.5
            )
            
            mask = proc_angles >= 5
            
            ff_line_norms = ff_proc[mask].tolist()
            fs_line_norms = fs_proc[mask].tolist()
            pts_line = pts_array[mask].tolist()
            angles_line = proc_angles[mask].tolist()

        if ff_line_norms:
            ff_out.append(np.vstack(ff_line_norms))
            fs_out.append(np.vstack(fs_line_norms))
            pts_out.append(np.vstack(pts_line))
            angles_out.append(np.array(angles_line))
        else:
            ff_out.append(np.empty((0, 3)))
            fs_out.append(np.empty((0, 3)))
            pts_out.append(np.empty((0, 3)))
            angles_out.append(np.empty(0))

        if angles_line:
            print(f"  completed loop {li+1}  – kept {len(pts_line):>3}/{len(line)} points"
                  f" (angles: {min(angles_line):.1f}°-{max(angles_line):.1f}°, "
                  f"mean {np.mean(angles_line):.1f}°)")
        else:
            print(f"  completed loop {li+1}  – kept {len(pts_line):>3}/{len(line)} points")

    return ff_out, fs_out, pts_out, angles_out

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
