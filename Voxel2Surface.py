import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure, filters, morphology, segmentation
from scipy.spatial import cKDTree
import trimesh


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