
import argparse
from pathlib import Path
from settings import *

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path: str):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    return image



def match_features(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    detector = cv2.SIFT_create()

    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("Could not detect enough image features.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    candidates12 = matcher.knnMatch(descriptors1, descriptors2, k=2)
    candidates21 = matcher.knnMatch(descriptors2, descriptors1, k=2)

    good12 = {
        first.queryIdx: first
        for pair in candidates12
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    }

    good21 = {
        first.queryIdx: first
        for pair in candidates21
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    }

    # Keep only mutual matches.
    good_matches = [
        match
        for query_idx, match in good12.items()
        if match.trainIdx in good21
        and good21[match.trainIdx].trainIdx == query_idx
    ]

    # Remove duplicate query/train index pairs.
    unique_matches = []
    seen = set()

    for match in good_matches:
        key = (match.queryIdx, match.trainIdx)

        if key not in seen:
            seen.add(key)
            unique_matches.append(match)

    good_matches = unique_matches

    # Remove matches whose keypoints are almost at the same
    # locations as already accepted matches in both images.
    spatial_matches = []

    for match in sorted(good_matches, key=lambda m: m.distance):
        p1 = keypoints1[match.queryIdx].pt
        p2 = keypoints2[match.trainIdx].pt

        duplicate = False

        for accepted in spatial_matches:
            q1 = keypoints1[accepted.queryIdx].pt
            q2 = keypoints2[accepted.trainIdx].pt

            if (
                np.linalg.norm(np.array(p1) - np.array(q1)) < 3
                and np.linalg.norm(np.array(p2) - np.array(q2)) < 3
            ):
                duplicate = True
                break

        if not duplicate:
            spatial_matches.append(match)

    good_matches = spatial_matches

    print(f"Keypoints: image1={len(keypoints1)}, image2={len(keypoints2)}")
    print(f"Ratio-test matches: image1->image2={len(good12)}, "
          f"image2->image1={len(good21)}")
    print(f"Mutual matches after deduplication: {len(good_matches)}")

    if len(good_matches) < 8:
        raise RuntimeError(
            f"Not enough mutual feature matches: {len(good_matches)}"
        )

    points1 = np.float32(
        [keypoints1[m.queryIdx].pt for m in good_matches]
    )
    points2 = np.float32(
        [keypoints2[m.trainIdx].pt for m in good_matches]
    )

    return points1, points2, good_matches


def reconstruct_points(image1, image2):
    points1, points2, matches = match_features(image1, image2)

    height, width = image1.shape[:2]

    # Approximate intrinsics for a prototype.
    # Replace with calibrated camera parameters for better accuracy.
    focal_length = float(max(width, height))
    principal_point = (width / 2.0, height / 2.0)

    camera_matrix = np.array([
        [focal_length, 0, principal_point[0]],
        [0, focal_length, principal_point[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    displacement = np.linalg.norm(points1 - points2, axis=1)
    large_motion = np.where(displacement > 20)[0]

    for i in large_motion:
        print(
            f"Match {i}: "
            f"p1={points1[i]}, "
            f"p2={points2[i]}, "
            f"displacement={displacement[i]:.2f}px"
        )

    essential, inlier_mask = cv2.findEssentialMat(
        points1,
        points2,
        camera_matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ESSENTIAL_MAT_THRESHOLD,
    )
    print(f"Feature matches: {len(matches)}")
    print(f"Geometric inliers: {np.count_nonzero(inlier_mask)}")

    homography, homography_mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        3.0,
    )

    print(
        "Homography inliers:",
        np.count_nonzero(homography_mask)
        if homography_mask is not None else 0
    )

    if essential is None:
        raise RuntimeError("Could not estimate the essential matrix.")

    # OpenCV may return multiple candidate essential matrices.
    essential = essential[:3, :3]

    inliers = inlier_mask.ravel().astype(bool)
    points1 = points1[inliers]
    points2 = points2[inliers]

    if len(points1) < 8:
        raise RuntimeError("Too few geometric inliers.")

    print("Essential shape:", essential.shape)
    print("Points1 shape:", points1.shape)
    print("Points2 shape:", points2.shape)
    print("Camera matrix:\n", camera_matrix)

    # Measure displacement between matched feature points.
    displacement = np.linalg.norm(points1 - points2, axis=1)

    print("Median displacement:", np.median(displacement))
    print("Mean displacement:", np.mean(displacement))
    print("Min displacement:", np.min(displacement))
    print("Max displacement:", np.max(displacement))
    print("Displacements:", np.round(displacement, 2))

    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        points1,
        points2,
        camera_matrix,
    )

    print("Pose inliers:", np.count_nonzero(pose_mask))
    print("Rotation:\n", rotation)
    print("Translation:\n", translation.ravel())

    # _, rotation, translation, pose_mask = cv2.recoverPose(
    #     essential,
    #     points1,
    #     points2,
    #     camera_matrix,
    # )
    # print(f"Pose inliers: {np.count_nonzero(pose_mask)}")

    pose_inliers = pose_mask.ravel() > 0
    points1 = points1[pose_inliers]
    points2 = points2[pose_inliers]

    normalized1 = cv2.undistortPoints(
        points1.reshape(-1, 1, 2), camera_matrix, None
    ).reshape(-1, 2)

    normalized2 = cv2.undistortPoints(
        points2.reshape(-1, 1, 2), camera_matrix, None
    ).reshape(-1, 2)

    projection1 = np.hstack([
        np.eye(3), np.zeros((3, 1))
    ])
    projection2 = np.hstack([rotation, translation])

    homogeneous = cv2.triangulatePoints(
        projection1,
        projection2,
        normalized1.T,
        normalized2.T,
    )

    points3d = (homogeneous[:3] / homogeneous[3]).T

    # Keep points in front of both estimated cameras.
    points_in_camera2 = (
        rotation @ points3d.T + translation
    ).T

    valid = (
        np.isfinite(points3d).all(axis=1)
        & (points3d[:, 2] > 0)
        & (points_in_camera2[:, 2] > 0)
    )

    points3d = points3d[valid]

    if len(points3d) == 0:
        raise RuntimeError("No valid 3D points were reconstructed.")

    return points3d, len(matches), int(valid.sum())


def save_ply(points: np.ndarray, output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")

        for x, y, z in points:
            file.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def show_point_cloud(points: np.ndarray):
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")

    axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=3,
    )

    axis.set_title("Reconstructor3D - Sparse Point Cloud")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")

    plt.tight_layout()
    plt.show()



def auto_crop_images(image1, image2):
    """Center-crop two images to the same dimensions."""
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    target_width = min(width1, width2)
    target_height = min(height1, height2)

    def crop_center(image, target_width, target_height):
        height, width = image.shape[:2]

        x_start = (width - target_width) // 2
        y_start = (height - target_height) // 2

        return image[
            y_start:y_start + target_height,
            x_start:x_start + target_width
        ]

    cropped1 = crop_center(image1, target_width, target_height)
    cropped2 = crop_center(image2, target_width, target_height)

    return cropped1, cropped2


def show_ply_file(ply_path):
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(ply_path))

    if cloud.is_empty():
        print(f"PLY contains no points: {ply_path}")
        return

    print(f"Opening 3D viewer: {ply_path}")
    print(f"Points displayed: {len(cloud.points)}")

    o3d.visualization.draw_geometries(
        [cloud],
        window_name=f"3D Point Cloud - {Path(ply_path).name}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct sparse 3D point clouds from multiple images."
    )
    parser.add_argument(
        "--input-dir",
        default="sample_images",
        help="Folder containing input images",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Folder for generated PLY files",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        help="Supported image extensions",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    pairs_dir = output_dir / "pairs"

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Image folder not found: {input_dir.resolve()}"
        )

    extensions = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in args.extensions
    }

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )

    if len(image_paths) < 2:
        raise RuntimeError(
            f"Need at least 2 images in {input_dir.resolve()}, "
            f"but found {len(image_paths)}."
        )

    pairs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input folder: {input_dir.resolve()}")
    print(f"Images found: {len(image_paths)}")
    for path in image_paths:
        print(f"  - {path.name}")

    total_pairs = len(image_paths) * (len(image_paths) - 1) // 2
    successful_pairs = 0
    failed_pairs = 0

    # Process every unique pair of images.
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            path1 = image_paths[i]
            path2 = image_paths[j]

            print("\n" + "=" * 60)
            print(f"Pair {successful_pairs + failed_pairs + 1}/{total_pairs}")
            print(f"Image 1: {path1.name}")
            print(f"Image 2: {path2.name}")

            try:
                image1 = load_image(str(path1))
                image2 = load_image(str(path2))

                if image1.shape[:2] != image2.shape[:2]:
                    print("Image dimensions do not match. Auto-cropping...")

                    image1, image2 = auto_crop_images(image1, image2)

                    print(
                        f"Images cropped to: "
                        f"{image1.shape[1]}x{image1.shape[0]}"
                    )

                points, match_count, point_count = reconstruct_points(
                    image1, image2
                )

                output_path = pairs_dir / (
                    f"{path1.stem}__{path2.stem}.ply"
                )

                save_ply(points, str(output_path))

                print(f"Feature matches: {match_count}")
                print(f"Valid 3D points: {point_count}")
                print(f"Point cloud saved to: {output_path}")

                successful_pairs += 1

            except (RuntimeError, ValueError, cv2.error) as error:
                failed_pairs += 1
                print(f"Pair failed: {error}")
                continue

    print("\n" + "=" * 60)
    print(f"Total image pairs: {total_pairs}")
    print(f"Successful pairs: {successful_pairs}")
    print(f"Failed pairs: {failed_pairs}")
    print(f"Output folder: {pairs_dir.resolve()}")

    if successful_pairs == 0:
        raise RuntimeError(
            "All image pairs failed. Check the image overlap "
            "and feature matching results."
        )

    ply_files = sorted(pairs_dir.glob("*.ply"))
    if ply_files:
        show_ply_file(ply_files[-1])


if __name__ == "__main__":
    main()
