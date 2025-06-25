import numpy as np
from scipy.spatial import cKDTree
import statsmodels.api as sm
from tqdm.auto import tqdm


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