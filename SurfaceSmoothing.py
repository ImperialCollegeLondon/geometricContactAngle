import numpy as np
import trimesh
from trimesh.smoothing import filter_taubin

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