"""
Accuracy benchmark: old_stitch.py (homography + RANSAC) vs stitch.py
(overlap-masked mutual SIFT + translation clustering).

Three test tiers:
  split      - existing *_left/_right split images (exact ground truth)
  synthetic  - crops of large photos with known shifts + perturbations
  real       - real camera pairs (no ground truth, overlap NCC proxy only)

Usage:
  .venv/bin/python compare_stitch.py [--tol 3] [--no-synthetic]
                                     [--save-images] [--seed 0]
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
BENCH_DIR = os.path.join(BASE_DIR, "outputs", "benchmark")

sys.path.insert(0, BASE_DIR)
import stitch  # noqa: E402


SPLIT_PAIRS = [
    ("solar_panel_left_35overlap.jpg", "solar_panel_right_35overlap.jpg"),
    ("solar_test_1_left_35overlap.jpg", "solar_test_1_right_35overlap.jpg"),
    ("solar_test_2_left_35overlap.jpg", "solar_test_2_right_35overlap.jpg"),
    ("solar_test_3_left_35overlap.jpg", "solar_test_3_right_35overlap.jpg"),
    ("solar_test_4_left_35overlap.jpg", "solar_test_4_right_35overlap.jpg"),
    ("demo_split_left_52overlap.jpg", "demo_split_right_52overlap.jpg"),
]

REAL_PAIRS = [
    ("x000_y000.jpg", "x051_y000.jpg"),
    ("2_x000_y000.jpg", "2_x051_y000.jpg"),
    ("3_x000_y000.jpg", "3_x051_y000.jpg"),
    ("4_x000_y000.jpg", "4_x051_y000.jpg"),
    ("5_x000_y000.jpg", "5_x071_y000.jpg"),
    ("6_x000_y000.jpg", "6_x051_y000.jpg"),
]

SYNTH_SOURCES = [
    "demo.jpg",
    "x000_y000.jpg",
    "2_x000_y000.jpg",
    "3_x000_y000.jpg",
    "4_x000_y000.jpg",
    "5_x000_y000.jpg",
    "6_x000_y000.jpg",
]

SYNTH_SOURCE_WIDTH = 1000
SYNTH_CROP_W = 500
SYNTH_CROP_H = 600
SYNTH_OVERLAPS = [0.20, 0.35, 0.50, 0.65, 0.80]
SYNTH_DY = [0, 6, -6]
SYNTH_PERTURBATIONS = ["none", "noise", "brightness", "blur", "rotate_1deg"]
ROTATE_PAD = 40


def translation_matrix(dx, dy):
    return np.array([
        [1.0, 0.0, dx],
        [0.0, 1.0, dy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# old_stitch.py lines 20-73, as a function

def old_estimate(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        raise RuntimeError("Not enough reliable matches.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    if H is None:
        raise RuntimeError("Could not estimate homography.")

    return H


# stitch.py, same path and overlap limits as process_image, minus debug images

def new_estimate(img1, img2):
    with contextlib.redirect_stdout(io.StringIO()):
        kp1, kp2, matches = stitch.find_mutual_sift_matches(
            img1, img2, "right", stitch.MAX_OVERLAP
        )
        H, _ = stitch.estimate_translation(
            img1, kp1, kp2, matches, "right",
            stitch.MIN_OVERLAP, stitch.MAX_OVERLAP
        )
    return H


def new_estimate_any_direction(img1, img2):
    """
    Oracle variant for real pairs: phone photos are not always oriented so
    that the x stage moves content "right", and process_image never passes
    its detected direction to stitch_pair. Try every direction and keep the
    alignment with the best overlap NCC.
    """
    best = None
    for direction in ("right", "left", "down", "up"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                kp1, kp2, matches = stitch.find_mutual_sift_matches(
                    img1, img2, direction, stitch.MAX_OVERLAP
                )
                H, _ = stitch.estimate_translation(
                    img1, kp1, kp2, matches, direction,
                    stitch.MIN_OVERLAP, stitch.MAX_OVERLAP
                )
        except Exception:
            continue
        ncc = overlap_ncc(img1, img2, H)[0]
        if best is None or ncc > best[0]:
            best = (ncc, H)
    if best is None:
        raise RuntimeError("No direction produced an alignment.")
    return best[1]


METHODS = [("old", old_estimate), ("new", new_estimate)]
REAL_EXTRA_METHODS = [("new_any_dir", new_estimate_any_direction)]


def corners_of(img):
    h, w = img.shape[:2]
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def corner_error(H_est, H_true, img2):
    """Mean distance (px) between img2's corners mapped by H_est and H_true."""
    c = corners_of(img2)
    est = cv2.perspectiveTransform(c, H_est).reshape(-1, 2)
    true = cv2.perspectiveTransform(c, H_true).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(est - true, axis=1)))


def overlap_ncc(img1, img2, H):
    """
    Warp img2 into img1's frame and score the overlap.
    Returns (ncc, mean_abs_diff, overlap_fraction_of_img1).
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    warped = cv2.warpPerspective(gray2, H, (w1, h1))
    mask = cv2.warpPerspective(np.ones((h2, w2), np.uint8), H, (w1, h1),
                               flags=cv2.INTER_NEAREST)
    # stay away from interpolated edge pixels
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0

    frac = float(mask.mean())
    if mask.sum() < 1000:
        return float("nan"), float("nan"), frac

    a = gray1[mask]
    b = warped[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    ncc = float((a * b).sum() / denom) if denom > 0 else float("nan")
    mad = float(np.abs(gray1[mask] - warped[mask]).mean())
    return ncc, mad, frac


def find_split_shift(left, right):
    """Exhaustive horizontal shift search; split pairs are exact crops."""
    L = left.astype(np.float32)
    R = right.astype(np.float32)
    w = L.shape[1]
    best_dx, best_diff = None, float("inf")
    for dx in range(int(w * 0.05), int(w * 0.95)):
        diff = np.abs(L[:, dx:] - R[:, :w - dx]).mean()
        if diff < best_diff:
            best_dx, best_diff = dx, diff
    return best_dx, best_diff


def load(name):
    img = cv2.imread(os.path.join(IMAGE_DIR, name))
    if img is None:
        raise RuntimeError(f"Could not load image: {name}")
    return img


# test pair builders. each yields dicts with img1, img2, H_true (or None)

def split_cases():
    for left_name, right_name in SPLIT_PAIRS:
        left = load(left_name)
        right = load(right_name)
        dx, diff = find_split_shift(left, right)
        yield {
            "tier": "split",
            "pair": left_name.replace("_left", ""),
            "perturbation": "none",
            "true_overlap": 1 - dx / left.shape[1],
            "img1": left,
            "img2": right,
            "H_true": translation_matrix(dx, 0),
        }


def perturb(img, kind, rng):
    if kind == "noise":
        noisy = img.astype(np.float32) + rng.normal(0, 6, img.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if kind == "brightness":
        return np.clip(img.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    if kind == "blur":
        return cv2.GaussianBlur(img, (5, 5), 0)
    return img


def synthetic_cases(rng):
    for source_name in SYNTH_SOURCES:
        src = load(source_name)
        scale = SYNTH_SOURCE_WIDTH / src.shape[1]
        src = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        H_src, W_src = src.shape[:2]

        x0 = ROTATE_PAD
        y0 = ROTATE_PAD + 10

        for overlap in SYNTH_OVERLAPS:
            dx = int(round(SYNTH_CROP_W * (1 - overlap)))
            if x0 + dx + SYNTH_CROP_W + ROTATE_PAD > W_src:
                continue

            for dy in SYNTH_DY:
                if y0 + dy + SYNTH_CROP_H + ROTATE_PAD > H_src:
                    continue

                img1 = src[y0:y0 + SYNTH_CROP_H, x0:x0 + SYNTH_CROP_W].copy()

                for kind in SYNTH_PERTURBATIONS:
                    x2, y2 = x0 + dx, y0 + dy

                    if kind == "rotate_1deg":
                        p = ROTATE_PAD
                        padded = src[y2 - p:y2 + SYNTH_CROP_H + p,
                                     x2 - p:x2 + SYNTH_CROP_W + p]
                        ph, pw = padded.shape[:2]
                        R = np.vstack([
                            cv2.getRotationMatrix2D((pw / 2, ph / 2), 1.0, 1.0),
                            [0, 0, 1],
                        ])
                        rotated = cv2.warpAffine(padded, R[:2], (pw, ph))
                        img2 = rotated[p:p + SYNTH_CROP_H, p:p + SYNTH_CROP_W].copy()
                        # img2 px -> rotated padded -> original padded -> img1
                        H_true = (
                            translation_matrix(dx - p, dy - p)
                            @ np.linalg.inv(R)
                            @ translation_matrix(p, p)
                        )
                    else:
                        img2 = src[y2:y2 + SYNTH_CROP_H, x2:x2 + SYNTH_CROP_W].copy()
                        img2 = perturb(img2, kind, rng)
                        H_true = translation_matrix(dx, dy)

                    yield {
                        "tier": "synthetic",
                        "pair": f"{source_name}_ov{int(overlap * 100)}_dy{dy:+d}",
                        "perturbation": kind,
                        "true_overlap": overlap,
                        "img1": img1,
                        "img2": img2,
                        "H_true": H_true,
                    }


def real_cases():
    for name1, name2 in REAL_PAIRS:
        yield {
            "tier": "real",
            "pair": f"{name1}->{name2}",
            "perturbation": "none",
            "true_overlap": float("nan"),
            "img1": load(name1),
            "img2": load(name2),
            "H_true": None,
        }


def save_composite(case, method, H):
    h1, w1 = case["img1"].shape[:2]
    warped = cv2.perspectiveTransform(corners_of(case["img2"]), H).reshape(-1, 2)
    span = np.ptp(np.vstack([warped, [[0, 0], [w1, h1]]]), axis=0)
    # a wild homography can produce an enormous canvas
    if span[0] * span[1] > 6 * w1 * h1:
        return
    with contextlib.redirect_stdout(io.StringIO()):
        result = stitch.feather_blend(case["img1"], case["img2"], H)
    safe = case["pair"].replace("/", "_").replace("->", "__")
    name = f"{case['tier']}_{safe}_{case['perturbation']}_{method}.jpg"
    cv2.imwrite(os.path.join(BENCH_DIR, name), result)


def run_case(case, tol, save_images):
    rows = []
    methods = METHODS + (REAL_EXTRA_METHODS if case["tier"] == "real" else [])
    for method, estimate in methods:
        row = {
            "tier": case["tier"],
            "pair": case["pair"],
            "perturbation": case["perturbation"],
            "true_overlap": case["true_overlap"],
            "method": method,
            "ok": False,
            "error": "",
            "corner_err_px": float("nan"),
            "corner_err_pct_width": float("nan"),
            "success": False,
            "est_dx": float("nan"),
            "est_dy": float("nan"),
            "ncc": float("nan"),
            "mad": float("nan"),
            "seconds": float("nan"),
        }

        start = time.perf_counter()
        try:
            H = estimate(case["img1"], case["img2"])
        except Exception as exc:
            row["seconds"] = time.perf_counter() - start
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        row["seconds"] = time.perf_counter() - start
        row["ok"] = True

        # translation at the centre of img2, comparable across both models
        h2, w2 = case["img2"].shape[:2]
        centre = np.float32([[[w2 / 2, h2 / 2]]])
        mapped = cv2.perspectiveTransform(centre, H).ravel()
        row["est_dx"] = float(mapped[0] - w2 / 2)
        row["est_dy"] = float(mapped[1] - h2 / 2)

        row["ncc"], row["mad"], _ = overlap_ncc(case["img1"], case["img2"], H)

        if case["H_true"] is not None:
            err = corner_error(H, case["H_true"], case["img2"])
            row["corner_err_px"] = err
            row["corner_err_pct_width"] = 100 * err / w2
            row["success"] = err <= tol

        if save_images:
            save_composite(case, method, H)

        rows.append(row)
    return rows


# reporting

def pct(x):
    return "  n/a" if x is None or np.isnan(x) else f"{x:6.1f}%"


def num(x, fmt="{:8.2f}"):
    return "     n/a" if x is None or np.isnan(x) else fmt.format(x)


def describe_change(old, new):
    """Human-readable change in an error value (lower is better)."""
    if np.isnan(old) or np.isnan(new):
        return "n/a"
    if old == new:
        return "same"
    if new < old:
        return f"{100 * (old - new) / old:.1f}% lower"
    if old == 0:
        return "higher"
    return f"{new / old:.1f}x higher"


def summarize_truth(rows, label, tol):
    """Summary for rows that have ground truth."""
    old = [r for r in rows if r["method"] == "old"]
    new = [r for r in rows if r["method"] == "new"]
    n = len(old)
    if n == 0:
        return None

    def rate(rs, pred):
        return 100 * sum(1 for r in rs if pred(r)) / len(rs)

    def within(limit):
        return lambda r: r["ok"] and r["corner_err_px"] <= limit

    old_succ, new_succ = rate(old, within(tol)), rate(new, within(tol))
    old_sub, new_sub = rate(old, within(1.0)), rate(new, within(1.0))
    old_fail = rate(old, lambda r: not r["ok"])
    new_fail = rate(new, lambda r: not r["ok"])

    # error stats only where both methods returned an estimate
    paired = [(o, w) for o, w in zip(old, new) if o["ok"] and w["ok"]]
    oe = np.array([o["corner_err_px"] for o, _ in paired])
    ne = np.array([w["corner_err_px"] for _, w in paired])

    better = int(np.sum(ne < oe - 0.5)) if len(paired) else 0
    worse = int(np.sum(ne > oe + 0.5)) if len(paired) else 0

    def stat(fn, arr):
        return float(fn(arr)) if len(arr) else float("nan")

    print(f"\n=== {label}  ({n} pairs) ===")
    print(f"{'':30}{'old':>9}{'new':>9}   change")
    print(f"{f'Success rate (<= {tol:g} px)':30}{pct(old_succ):>9}{pct(new_succ):>9}"
          f"   {new_succ - old_succ:+.1f} pp")
    print(f"{'Sub-pixel rate (<= 1 px)':30}{pct(old_sub):>9}{pct(new_sub):>9}"
          f"   {new_sub - old_sub:+.1f} pp")
    print(f"{'Failure rate (no result)':30}{pct(old_fail):>9}{pct(new_fail):>9}"
          f"   {new_fail - old_fail:+.1f} pp")
    if len(paired):
        for name, fn in [("Median corner error (px)", np.median),
                         ("Mean corner error (px)", np.mean),
                         ("90th pct corner error (px)", lambda a: np.percentile(a, 90))]:
            o, w = stat(fn, oe), stat(fn, ne)
            print(f"{name:30}{num(o, '{:9.3f}'):>9}{num(w, '{:9.3f}'):>9}"
                  f"   {describe_change(o, w)}")
        print(f"{'New better / worse (>0.5 px)':30}{better:>9}{worse:>9}"
              f"   of {len(paired)} pairs both produced a result")
    o_t = np.median([r["seconds"] for r in old])
    w_t = np.median([r["seconds"] for r in new])
    print(f"{'Median runtime (s)':30}{num(o_t, '{:9.3f}'):>9}{num(w_t, '{:9.3f}'):>9}")

    return {
        "old_succ": old_succ, "new_succ": new_succ,
        "old_sub": old_sub, "new_sub": new_sub,
        "old_median": stat(np.median, oe), "new_median": stat(np.median, ne),
        "old_mean": stat(np.mean, oe), "new_mean": stat(np.mean, ne),
    }


def summarize_real(rows):
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    if not by_method:
        return
    methods = list(by_method)
    pairs = [r["pair"] for r in by_method[methods[0]]]

    print(f"\n=== Real camera pairs ({len(pairs)} pairs, no ground truth) ===")
    print("Proxy: NCC of the overlap after alignment (1.0 = perfect), higher is better.")
    print("new_any_dir = new code tried in all four directions (best NCC kept).")
    header = "".join(f"{m + ' NCC':>22}" for m in methods)
    print(f"{'pair':36}{header}")
    for i, pair in enumerate(pairs):
        cells = []
        for m in methods:
            r = by_method[m][i]
            if r["ok"]:
                cells.append(f"{r['ncc']:.3f} ({r['est_dx']:+.0f},{r['est_dy']:+.0f})")
            else:
                cells.append("FAIL")
        print(f"{pair[:36]:36}" + "".join(f"{c:>22}" for c in cells))

    def mean_ncc(rs):
        vals = [r["ncc"] for r in rs if r["ok"] and not np.isnan(r["ncc"])]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"{'Mean NCC (successful only)':36}"
          + "".join(f"{num(mean_ncc(by_method[m]), '{:.3f}'):>22}" for m in methods))
    print(f"{'Failures':36}"
          + "".join(f"{sum(not r['ok'] for r in by_method[m]):>22}" for m in methods))
    print("(numbers in brackets: estimated dx, dy shift of image 2 in px)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tol", type=float, default=3.0,
                        help="corner error (px) that still counts as a success")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="skip the generated synthetic pairs")
    parser.add_argument("--save-images", action="store_true",
                        help="write each method's stitched result to outputs/benchmark")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(BENCH_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    generators = [split_cases()]
    if not args.no_synthetic:
        generators.append(synthetic_cases(rng))
    generators.append(real_cases())

    rows = []
    count = 0
    for gen in generators:
        for case in gen:
            count += 1
            print(f"\r[{count:4d}] {case['tier']:9} {case['pair'][:40]:40} "
                  f"{case['perturbation']:12}", end="", flush=True)
            rows.extend(run_case(case, args.tol, args.save_images))
    print()

    csv_path = os.path.join(BENCH_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCorner error = mean distance between where image 2's corners land "
          f"and where they should land.")

    tol = args.tol
    split = [r for r in rows if r["tier"] == "split"]
    synth = [r for r in rows if r["tier"] == "synthetic"]
    synth_shift = [r for r in synth if r["perturbation"] != "rotate_1deg"]
    synth_rot = [r for r in synth if r["perturbation"] == "rotate_1deg"]

    summarize_truth(split, "Split pairs (exact ground truth)", tol)
    for kind in SYNTH_PERTURBATIONS:
        summarize_truth([r for r in synth if r["perturbation"] == kind],
                        f"Synthetic: {kind}", tol)
    for ov in SYNTH_OVERLAPS:
        summarize_truth([r for r in synth_shift if r["true_overlap"] == ov],
                        f"Synthetic: {int(ov * 100)}% overlap (no rotation)", tol)

    headline = summarize_truth(split + synth_shift,
                               "HEADLINE: pure-translation pairs (split + synthetic, "
                               "no rotation)", tol)
    rotation = summarize_truth(synth_rot, "STRESS TEST: 1 degree camera rotation", tol)
    summarize_real([r for r in rows if r["tier"] == "real"])

    if headline:
        print("\n--- Improvement summary (pure-translation pairs) ---")
        print(f"Success rate (<= {tol:g} px): {headline['old_succ']:.1f}% -> "
              f"{headline['new_succ']:.1f}% ({headline['new_succ'] - headline['old_succ']:+.1f} pp)")
        print(f"Sub-pixel rate (<= 1 px): {headline['old_sub']:.1f}% -> "
              f"{headline['new_sub']:.1f}% ({headline['new_sub'] - headline['old_sub']:+.1f} pp)")
        print(f"Median error: {headline['old_median']:.3f} px -> {headline['new_median']:.3f} px "
              f"({describe_change(headline['old_median'], headline['new_median'])})")
        print(f"Mean error:   {headline['old_mean']:.3f} px -> {headline['new_mean']:.3f} px "
              f"({describe_change(headline['old_mean'], headline['new_mean'])})")
    if rotation:
        print(f"With 1 deg rotation: success {rotation['old_succ']:.1f}% -> "
              f"{rotation['new_succ']:.1f}%, median error {rotation['old_median']:.2f} -> "
              f"{rotation['new_median']:.2f} px (translation-only model can't rotate)")

    print(f"\nPer-pair results: {csv_path}")


if __name__ == "__main__":
    main()
