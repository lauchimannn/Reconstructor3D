
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def find_colmap(custom_path=None):
    if custom_path:
        path = Path(custom_path)
        if path.is_file():
            return str(path.resolve())

        found = shutil.which(custom_path)
        if found:
            return found

    # Search in the local COLMAP folder
    base_dir = Path(__file__).resolve().parent
    colmap_dir = base_dir / "colmap-x64-windows-cuda"

    for name in ("COLMAP.bat", "COLMAP.exe", "colmap.exe", "colmap"):
        candidate = colmap_dir / name
        if candidate.is_file():
            return str(candidate.resolve())

    # Search in system PATH
    for name in ("colmap", "colmap.exe", "COLMAP.bat", "COLMAP.exe"):
        found = shutil.which(name)
        if found:
            return found

    return None


def run_colmap(executable, *arguments):
    command = [executable, *map(str, arguments)]

    print("\n>", subprocess.list2cmdline(command), flush=True)

    # Windows .bat/.cmd launchers require a command shell.
    use_shell = executable.lower().endswith((".bat", ".cmd"))

    subprocess.run(command, check=True, shell=use_shell)


def find_sparse_model(sparse_dir):
    if not sparse_dir.exists():
        return None

    models = [
        path for path in sparse_dir.iterdir()
        if path.is_dir()
        and (path / "cameras.bin").exists()
        and (path / "images.bin").exists()
        and (path / "points3D.bin").exists()
    ]

    if not models:
        return None

    return max(
        models,
        key=lambda path: (path / "points3D.bin").stat().st_size,
    )


def open_point_cloud(ply_path):
    try:
        import open3d as o3d
    except ImportError:
        print(
            "\nTo open the PLY automatically, install Open3D:\n"
            "python -m pip install open3d"
        )
        return

    cloud = o3d.io.read_point_cloud(str(ply_path))

    if cloud.is_empty():
        print(f"Point cloud is empty: {ply_path}")
        return

    print(f"Opening point cloud: {len(cloud.points)} points")

    o3d.visualization.draw_geometries(
        [cloud],
        window_name="COLMAP Dense Point Cloud",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-view 3D reconstruction using COLMAP."
    )
    parser.add_argument(
        "--input-dir",
        default="sample_images",
        help="Folder containing all input photos",
    )
    parser.add_argument(
        "--output-dir",
        default="output/colmap",
        help="Output folder",
    )
    parser.add_argument(
        "--colmap",
        default=None,
        help="Optional full path to the COLMAP executable",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=2000,
        help="Maximum image size for dense reconstruction",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Do not open the viewer automatically",
    )
    args = parser.parse_args()

    image_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image folder does not exist: {image_dir}"
        )

    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(images) < 3:
        raise RuntimeError(
            f"Found only {len(images)} images. "
            "Use at least 3 overlapping views."
        )

    colmap = find_colmap(args.colmap)

    if colmap is None:
        raise FileNotFoundError(
            "COLMAP was not found. Install COLMAP first, then add it "
            "to PATH or provide its executable path using --colmap."
        )

    # Refuse to overwrite an existing reconstruction.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output folder already contains files: {output_dir}\n"
            "Choose another folder with --output-dir."
        )

    database = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    dense_dir = output_dir / "dense"

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    print(f"COLMAP: {colmap}")
    print(f"Images found: {len(images)}")

    for image in images:
        print(f"  {image.name}")

    # 1. Extract SIFT features from every image.
    run_colmap(
        colmap,
        "feature_extractor",
        "--database_path", database,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_num_features", "8192",
    )

    # 2. Match features across image pairs.
    run_colmap(
        colmap,
        "exhaustive_matcher",
        "--database_path", database,
    )

    # 3. Estimate camera poses and create a common sparse model.
    run_colmap(
        colmap,
        "mapper",
        "--database_path", database,
        "--image_path", image_dir,
        "--output_path", sparse_dir,
    )

    sparse_model = find_sparse_model(sparse_dir)

    if sparse_model is None:
        raise RuntimeError(
            "COLMAP could not reconstruct a connected camera model. "
            "Check image overlap, blur, repeated textures and lighting."
        )

    print(f"\nSparse model: {sparse_model}")

    # 4. Undistort images for dense reconstruction.
    dense_dir.mkdir(parents=True, exist_ok=True)

    run_colmap(
        colmap,
        "image_undistorter",
        "--image_path", image_dir,
        "--input_path", sparse_model,
        "--output_path", dense_dir,
        "--output_type", "COLMAP",
        "--max_image_size", args.max_image_size,
    )

    # 5. Compute depth maps.
    run_colmap(
        colmap,
        "patch_match_stereo",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    )

    # 6. Fuse depth maps into a colored dense point cloud.
    dense_ply = dense_dir / "fused.ply"

    run_colmap(
        colmap,
        "stereo_fusion",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", dense_ply,
    )

    if not dense_ply.exists() or dense_ply.stat().st_size == 0:
        raise RuntimeError(
            "COLMAP did not create the dense point cloud."
        )

    print(f"\nDense point cloud saved: {dense_ply}")

    # 7. Attempt to reconstruct a surface mesh.
    mesh_ply = dense_dir / "meshed-poisson.ply"

    try:
        run_colmap(
            colmap,
            "poisson_mesher",
            "--input_path", dense_ply,
            "--output_path", mesh_ply,
        )

        if mesh_ply.exists():
            print(f"Surface mesh saved: {mesh_ply}")

    except subprocess.CalledProcessError as error:
        print(
            "\nWarning: point cloud was created, but mesh generation "
            f"failed (exit code {error.returncode})."
        )

    if not args.no_viewer:
        open_point_cloud(dense_ply)

    print("\nReconstruction finished.")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
