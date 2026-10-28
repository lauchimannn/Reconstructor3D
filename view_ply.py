
import open3d as o3d
from pathlib import Path

# ply_path = Path("output/colmap/dense/fused.ply")
# ply_path = Path("output/point_cloud.ply")
ply_path = Path("output/colmap/dense/meshed-poisson.ply")

if not ply_path.exists():
    raise FileNotFoundError(f"PLY file not found: {ply_path.resolve()}")

# Load the PLY file
geometry = o3d.io.read_triangle_mesh(str(ply_path))

if geometry.has_triangles():
    print("Type: Mesh")
    print(f"Vertices: {len(geometry.vertices):,}")
    print(f"Triangles: {len(geometry.triangles):,}")

    if not geometry.has_vertex_normals():
        geometry.compute_vertex_normals()
else:
    geometry = o3d.io.read_point_cloud(str(ply_path))
    print("Type: Point Cloud")
    print(f"Points: {len(geometry.points):,}")

if geometry.is_empty():
    raise ValueError("The PLY file contains no usable geometry.")

# Display the 3D model
o3d.visualization.draw_geometries(
    [geometry],
    window_name="COLMAP 3D Reconstruction",
    width=1280,
    height=800
)
