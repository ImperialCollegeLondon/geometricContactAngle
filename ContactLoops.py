import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.interpolate import LSQUnivariateSpline
import trimesh

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