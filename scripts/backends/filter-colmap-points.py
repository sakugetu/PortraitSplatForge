import argparse
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_QUANTILE = {
    "draft": 0.955,
    "balanced": 0.975,
    "high": 0.988,
}

DEFAULT_MAX_POINTS = {
    "draft": 80000,
    "balanced": 140000,
    "high": 220000,
}


def parse_point(line):
    parts = line.split()
    if len(parts) < 8:
        return None
    try:
        return {
            "id": int(parts[0]),
            "xyz": np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64),
            "rgb": np.array([int(parts[4]), int(parts[5]), int(parts[6])], dtype=np.int32),
            "error": float(parts[7]),
            "line": line,
        }
    except ValueError:
        return None


def robust_keep_mask(points, quality, quantile, max_points):
    xyz = np.stack([point["xyz"] for point in points])
    finite = np.isfinite(xyz).all(axis=1)
    if finite.sum() < 16:
        return finite

    center = np.median(xyz[finite], axis=0)
    radius = np.linalg.norm(xyz - center, axis=1)
    finite_radius = radius[np.isfinite(radius)]
    radius_cut = np.quantile(finite_radius, quantile) if finite_radius.size else math.inf

    errors = np.array([point["error"] for point in points], dtype=np.float64)
    finite_error = errors[np.isfinite(errors)]
    error_cut = np.quantile(finite_error, 0.985 if quality == "high" else 0.975) if finite_error.size else math.inf

    keep = finite & np.isfinite(radius) & (radius <= radius_cut) & (errors <= max(error_cut, 2.0))

    if keep.sum() > max_points:
        kept_indices = np.where(keep)[0]
        order = np.lexsort((errors[kept_indices], radius[kept_indices]))
        selected = kept_indices[order[:max_points]]
        limited = np.zeros_like(keep, dtype=bool)
        limited[selected] = True
        keep = limited

    return keep


def rewrite_images(images_path, removed_ids):
    if not images_path.exists() or not removed_ids:
        return
    lines = images_path.read_text(encoding="utf8").splitlines()
    rewritten = []
    data_line_index = 0
    for line in lines:
        if line.startswith("#") or not line.strip():
            rewritten.append(line)
            continue
        is_points_line = data_line_index % 2 == 1
        data_line_index += 1
        if not is_points_line:
            rewritten.append(line)
            continue
        tokens = line.split()
        for token_index in range(2, len(tokens), 3):
            try:
                point_id = int(tokens[token_index])
            except ValueError:
                continue
            if point_id in removed_ids:
                tokens[token_index] = "-1"
        rewritten.append(" ".join(tokens))
    images_path.write_text("\n".join(rewritten) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sparse_dir")
    parser.add_argument("--quality", default="balanced")
    parser.add_argument("--quantile", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--output-stats", default=None)
    args = parser.parse_args()

    sparse_dir = Path(args.sparse_dir)
    points_path = sparse_dir / "points3D.txt"
    images_path = sparse_dir / "images.txt"
    if not points_path.exists():
        raise FileNotFoundError(points_path)

    comments = []
    point_lines = []
    for line in points_path.read_text(encoding="utf8").splitlines():
        if line.startswith("#") or not line.strip():
            comments.append(line)
            continue
        point = parse_point(line)
        if point is not None:
            point_lines.append(point)

    if not point_lines:
        return

    quantile = args.quantile or DEFAULT_QUANTILE.get(args.quality, DEFAULT_QUANTILE["balanced"])
    max_points = args.max_points or DEFAULT_MAX_POINTS.get(args.quality, DEFAULT_MAX_POINTS["balanced"])
    keep = robust_keep_mask(point_lines, args.quality, quantile, max_points)
    kept = [point for point, should_keep in zip(point_lines, keep) if should_keep]
    removed_ids = {point["id"] for point, should_keep in zip(point_lines, keep) if not should_keep}

    backup_path = sparse_dir / "points3D.unfiltered.txt"
    if not backup_path.exists():
        backup_path.write_text(points_path.read_text(encoding="utf8"), encoding="utf8")

    output_lines = comments + [point["line"] for point in kept]
    points_path.write_text("\n".join(output_lines) + "\n", encoding="utf8")
    rewrite_images(images_path, removed_ids)

    stats = {
        "inputPoints": len(point_lines),
        "keptPoints": len(kept),
        "removedPoints": len(removed_ids),
        "quantile": quantile,
        "maxPoints": max_points,
        "quality": args.quality,
    }
    if args.output_stats:
        Path(args.output_stats).write_text(json.dumps(stats, indent=2), encoding="utf8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
