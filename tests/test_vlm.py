"""
VLM Object Detection — Accuracy Evaluation on COCO 2017

Evaluation protocol (per project specification Section V-A):
  - 100 indoor scene images from COCO 2017 val split
  - Categories: sofa/couch, chair, dining table, bed, toilet, tv, sink
  - Correct if predicted bounding box IoU >= 0.5 with any COCO GT annotation
  - Pass threshold: >= 75% accuracy

Usage:
    python3 tests/test_vlm.py --coco-dir coco/
    python3 tests/test_vlm.py --coco-dir coco/ --samples 100 --threshold 0.5
"""

import sys, os, argparse, time, json, io, contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vlm.grounding_dino import detect_object, load_model
from PIL import Image

# ── Indoor categories to evaluate (COCO category_id → query text) ─────────────
INDOOR_CATEGORIES = {
    62: "chair",
    63: "couch",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    81: "sink",
    82: "refrigerator",
}

# Target samples per category (adjust to reach ~100 total)
SAMPLES_PER_CAT = {
    62: 20,   # chair
    63: 15,   # couch
    65: 15,   # bed
    67: 20,   # dining table
    70: 10,   # toilet
    72: 10,   # tv
    81:  5,   # sink
    82:  5,   # refrigerator
}   # total = 100


# ── IoU ───────────────────────────────────────────────────────────────────────
def _iou(pred_box: list, gt_box_xywh: list) -> float:
    """
    pred_box   : [x1, y1, x2, y2]  (Grounding DINO output)
    gt_box_xywh: [x, y, w, h]      (COCO annotation format)
    """
    gx, gy, gw, gh = gt_box_xywh
    gt = [gx, gy, gx + gw, gy + gh]

    ix1 = max(pred_box[0], gt[0])
    iy1 = max(pred_box[1], gt[1])
    ix2 = min(pred_box[2], gt[2])
    iy2 = min(pred_box[3], gt[3])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    area_gt   = gw * gh
    union     = area_pred + area_gt - inter

    return inter / union if union > 0 else 0.0


# ── Load COCO and select 100 samples ─────────────────────────────────────────
def load_coco_samples(coco_dir: str) -> list[dict]:
    ann_path = os.path.join(coco_dir, "annotations", "instances_val2017.json")
    img_dir  = os.path.join(coco_dir, "images", "val2017")

    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    # Build lookup: image_id → file_name
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}

    # Build lookup: category_id → list of annotations
    cat2anns: dict[int, list] = {cid: [] for cid in INDOOR_CATEGORIES}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid in cat2anns:
            cat2anns[cid].append(ann)

    samples = []
    for cid, query in INDOOR_CATEGORIES.items():
        anns      = cat2anns[cid]
        seen_imgs = set()
        count     = 0
        for ann in anns:
            if count >= SAMPLES_PER_CAT[cid]:
                break
            iid  = ann["image_id"]
            if iid in seen_imgs:
                continue
            fpath = os.path.join(img_dir, id2file[iid])
            if not os.path.exists(fpath):
                continue
            # Collect all GT boxes for this category in this image
            gt_boxes = [a["bbox"] for a in anns if a["image_id"] == iid]
            samples.append({
                "image_id":    iid,
                "image_path":  fpath,
                "category_id": cid,
                "query":       query,
                "gt_boxes":    gt_boxes,
            })
            seen_imgs.add(iid)
            count += 1

    return samples


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(samples: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    results = []
    for i, s in enumerate(samples):
        image      = Image.open(s["image_path"]).convert("RGB")
        query      = s["query"]
        gt_boxes   = s["gt_boxes"]

        t0   = time.time()
        pred = detect_object(image, query)
        elapsed = time.time() - t0

        if not pred.get("found"):
            best_iou = 0.0
            correct  = False
        else:
            pred_box = pred["box"]   # [x1, y1, x2, y2]
            best_iou = max(_iou(pred_box, gt) for gt in gt_boxes)
            correct  = best_iou >= iou_threshold

        results.append({
            "id":         i + 1,
            "image_id":   s["image_id"],
            "query":      query,
            "found":      pred.get("found", False),
            "score":      pred.get("score", 0.0),
            "pred_box":   pred.get("box"),
            "best_iou":   round(best_iou, 3),
            "correct":    correct,
            "latency_s":  round(elapsed, 3),
        })
    return results


# ── Build report text ─────────────────────────────────────────────────────────
def build_report(results: list[dict], mode: str,
                 iou_thr: float, threshold: float = 0.75) -> str:
    total     = len(results)
    n_correct = sum(r["correct"] for r in results)
    accuracy  = n_correct / total
    avg_lat   = sum(r["latency_s"] for r in results) / total
    passed    = accuracy >= threshold

    # Per-category breakdown
    from collections import defaultdict
    cat_stats: dict[str, list] = defaultdict(list)
    for r in results:
        cat_stats[r["query"]].append(r["correct"])

    lines = []
    sep = "=" * 72
    lines += [
        sep,
        "TEST: Grounding DINO Object Detection — COCO 2017 Evaluation",
        f"Date         : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode         : {mode}",
        f"Dataset      : COCO 2017 val2017",
        f"IoU threshold: >= {iou_thr}",
        f"Cases        : {total}",
        f"Correct      : {n_correct} / {total}",
        f"Accuracy     : {accuracy * 100:.1f}%   (pass threshold: {threshold * 100:.0f}%)",
        f"Result       : {'PASS' if passed else 'FAIL'}",
        f"Avg latency  : {avg_lat:.3f} s / sample",
        sep,
        "",
        "PER-CATEGORY BREAKDOWN:",
        f"  {'Category':<16}  {'Correct':>8}  {'Total':>6}  {'Accuracy':>9}",
        f"  {'-'*16}  {'-'*8}  {'-'*6}  {'-'*9}",
    ]
    for cat, vals in sorted(cat_stats.items()):
        nc = sum(vals)
        nt = len(vals)
        lines.append(f"  {cat:<16}  {nc:>8}  {nt:>6}  {nc/nt*100:>8.1f}%")

    failures = [r for r in results if not r["correct"]]
    if failures:
        lines += [
            "",
            f"FAILED ({len(failures)} / {total}):",
            f"  {'#':>4}  {'Query':<16}  {'Found':>6}  {'Score':>6}  {'IoU':>6}",
            f"  {'-'*4}  {'-'*16}  {'-'*6}  {'-'*6}  {'-'*6}",
        ]
        for r in failures:
            lines.append(
                f"  {r['id']:>4}  {r['query']:<16}  "
                f"{'Yes' if r['found'] else 'No':>6}  "
                f"{r['score']:>6.3f}  {r['best_iou']:>6.3f}"
            )

    lines += ["", sep, "", "DETAIL (all cases):", ""]
    for r in results:
        status = "PASS" if r["correct"] else "FAIL"
        lines += [
            f"[{r['id']:03d}] {status}",
            f"  Image ID  : {r['image_id']}",
            f"  Query     : {r['query']}",
            f"  Found     : {r['found']}",
            f"  Score     : {r['score']:.3f}",
            f"  Pred box  : {r['pred_box']}",
            f"  Best IoU  : {r['best_iou']:.3f}",
            f"  Latency   : {r['latency_s']} s",
            "",
        ]
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import contextlib, io

    parser = argparse.ArgumentParser(
        description="Evaluate Grounding DINO on COCO 2017 indoor scenes"
    )
    parser.add_argument(
        "--coco-dir", type=str, required=True,
        help="Path to coco/ directory containing images/ and annotations/"
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Total number of test samples (default: 100)"
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="IoU threshold for correct detection (default: 0.5)"
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir      = os.path.join(project_root, "test_log")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    log_path  = os.path.join(log_dir, f"coco_val2017_{timestamp}.log")

    sys.stdout.write(f"Running VLM evaluation — output → {log_path}\n")
    sys.stdout.flush()

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        samples = load_coco_samples(args.coco_dir)
        print(f"[COCO] Loaded {len(samples)} samples across {len(INDOOR_CATEGORIES)} categories")
        load_model()   # preload Grounding DINO once
        results = evaluate(samples, iou_threshold=args.iou_threshold)

    loader_output = captured.getvalue().splitlines()
    report = build_report(
        results,
        mode=f"Grounding DINO (grounding-dino-tiny)  IoU>={args.iou_threshold}",
        iou_thr=args.iou_threshold,
    )

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(loader_output) + "\n\n")
        f.write(report)

    total     = len(results)
    n_correct = sum(r["correct"] for r in results)
    sys.stdout.write(
        f"Done — {n_correct}/{total} correct ({n_correct/total*100:.1f}%)  |  "
        f"log: {log_path}\n"
    )
