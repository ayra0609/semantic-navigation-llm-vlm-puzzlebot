"""
LLM Intent Extraction — Accuracy Evaluation on ALFRED Dataset

Evaluation protocol (per project specification Section V-A):
  - 100 navigation instructions from ALFRED dataset (GotoLocation steps only)
  - Correct if extracted target label matches ALFRED ground-truth object label
  - Pass threshold: >= 85% target label accuracy

Usage:
    python tests/test_intent.py --alfred-path <path/to/alfred/data/json_2.1.0>
    python tests/test_intent.py --alfred-path <...> --split valid_seen
    python tests/test_intent.py --alfred-path <...> --regex-only

ALFRED data is already at:
    alfred/data/json_2.1.0/
"""

import sys, os, argparse, time, json, glob, re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.intent_extraction import extract_navigation_intent, _regex_extract


# ── Load cases from real ALFRED traj_data.json files ─────────────────────────
def load_alfred_cases(alfred_dir: str, split: str = "valid_seen",
                      max_cases: int = 100) -> list[dict]:
    """
    Walk alfred_dir/<split>/**/traj_data.json and extract (instruction, gt_label)
    pairs for every GotoLocation step.

    ALFRED annotation structure:
        turk_annotations.anns[0].high_descs[step_idx]       <- instruction text
        plan.high_pddl[step_idx].discrete_action.action      <- "GotoLocation"
        plan.high_pddl[step_idx].discrete_action.args[0]     <- GT object label
    """
    split_dir = os.path.join(alfred_dir, split)
    if not os.path.isdir(split_dir):
        split_dir = alfred_dir

    pattern    = os.path.join(split_dir, "**", "traj_data.json")
    json_files = glob.glob(pattern, recursive=True)

    if not json_files:
        raise FileNotFoundError(
            f"No traj_data.json files found under '{split_dir}'.\n"
            f"Expected path: alfred/data/json_2.1.0/{split}/"
        )

    raw     = []
    skipped = 0

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            anns = data.get("turk_annotations", {}).get("anns", [])
            pddl = data.get("plan", {}).get("high_pddl", [])

            if not anns:
                skipped += 1
                continue

            ann0 = anns[0]

            for step_idx, step in enumerate(pddl):
                da = step.get("discrete_action", {})
                if da.get("action") != "GotoLocation":
                    continue

                args = da.get("args", [])
                if not args:
                    skipped += 1
                    continue

                gt_label = args[0].strip().lower()

                descs = ann0.get("high_descs", [])
                if step_idx >= len(descs):
                    skipped += 1
                    continue

                instr = descs[step_idx].strip()
                if instr:
                    raw.append({"instruction": instr, "gt": gt_label})

        except (json.JSONDecodeError, KeyError, IndexError):
            skipped += 1
            continue

    # Filter: only keep cases where GT object name appears in the instruction text.
    # This removes cases where the human description and AI2-THOR label are
    # semantically misaligned (e.g. "go to the table" → GT "dresser").
    cases = [c for c in raw if _gt_in_instruction(c["instruction"], c["gt"])]
    cases = cases[:max_cases]

    print(f"[ALFRED] Raw GotoLocation steps : {len(raw)}")
    print(f"[ALFRED] After alignment filter : {len(cases)}  (skipped {skipped} bad files)")
    return cases


def _gt_in_instruction(instruction: str, gt: str) -> bool:
    """
    Return True if the GT object label (or its space-separated form) appears
    in the instruction, so we know the human explicitly named the target.
    E.g.  gt='coffeetable'  matches  'coffee table'  or  'coffeetable'.
    """
    instr_flat = instruction.lower().replace(" ", "")
    gt_flat    = gt.strip().lower().replace(" ", "")
    return gt_flat in instr_flat


# ── Evaluation ────────────────────────────────────────────────────────────────
_ARTICLES = re.compile(r'\b(the|a|an)\b')

def _normalise(s) -> str:
    """
    Lowercase, strip leading/trailing articles, remove spaces.
    'the desk lamp' → 'desklamp',  'a coffee table' → 'coffeetable'
    Handles cases where LLM returns a list instead of a string.
    """
    if isinstance(s, list):
        s = " ".join(str(x) for x in s)
    s = str(s).strip().lower()
    s = _ARTICLES.sub('', s)
    return s.replace(" ", "").strip()


def evaluate(cases: list[dict], use_ollama: bool) -> list[dict]:
    """Run each instruction through the extractor and record raw LLM output + result."""
    extractor = extract_navigation_intent if use_ollama else _regex_extract
    results   = []
    for i, case in enumerate(cases):
        instr   = case["instruction"]
        gt      = _normalise(case["gt"])
        t0      = time.time()
        pred    = extractor(instr) or {}
        elapsed = time.time() - t0
        pred_target = _normalise(pred.get("target", ""))
        # Lenient match: exact OR containment in either direction
        # e.g. gt='desklamp', pred='lamponthedesk' → gt in pred → correct
        correct = (pred_target == gt) or (gt in pred_target) or (pred_target in gt and pred_target != "")
        results.append({
            "id":          i + 1,
            "instruction": instr,
            "gt":          case["gt"],
            "gt_norm":     gt,
            "llm_raw":     pred,
            "pred":        pred_target,
            "correct":     correct,
            "latency_s":   round(elapsed, 3),
        })
    return results


# ── Log saving ────────────────────────────────────────────────────────────────
def build_report_text(results: list[dict], mode: str,
                      split: str, threshold: float = 0.85) -> str:
    """Return the full report as a string (written to log, not printed)."""
    total     = len(results)
    n_correct = sum(r["correct"] for r in results)
    accuracy  = n_correct / total
    avg_lat   = sum(r["latency_s"] for r in results) / total
    passed    = accuracy >= threshold

    lines = []
    sep = "=" * 72
    lines += [
        sep,
        "TEST: LLM Intent Extraction — ALFRED Evaluation",
        f"Date         : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mode         : {mode}",
        f"Split        : {split}",
        f"Cases        : {total}",
        f"Filter       : GT label present in instruction text",
        f"Match rule   : exact OR substring containment (articles stripped)",
        f"Correct      : {n_correct} / {total}",
        f"Accuracy     : {accuracy * 100:.1f}%   (pass threshold: {threshold * 100:.0f}%)",
        f"Result       : {'PASS' if passed else 'FAIL'}",
        f"Avg latency  : {avg_lat:.3f} s / sample",
        sep, "",
    ]

    failures = [r for r in results if not r["correct"]]
    if failures:
        lines.append(f"FAILED ({len(failures)} / {total}):")
        lines.append(f"  {'#':>4}  {'Instruction':<50}  {'GT':>16}  {'Predicted':>16}")
        lines.append(f"  {'-'*4}  {'-'*50}  {'-'*16}  {'-'*16}")
        for r in failures:
            instr = (r["instruction"][:48] + "..") if len(r["instruction"]) > 50 else r["instruction"]
            lines.append(f"  {r['id']:>4}  {instr:<50}  {r['gt']:>16}  {r['pred']:>16}")
        lines.append("")

    lines += [sep, "", "DETAIL (all cases):", ""]
    for r in results:
        status = "PASS" if r["correct"] else "FAIL"
        lines += [
            f"[{r['id']:03d}] {status}",
            f"  Instruction : {r['instruction']}",
            f"  GT label    : {r['gt']}",
            f"  LLM output  : {json.dumps(r['llm_raw'])}",
            f"  Predicted   : {r['pred']}",
            f"  Latency     : {r['latency_s']} s",
            "",
        ]

    return "\n".join(lines)


def save_log(results: list[dict], mode: str, split: str,
             log_dir: str, extra_lines: list[str] | None = None) -> str:
    """
    Write full report to test_log/alfred_<split>_<YYYYMMDDHHMMSS>.log
    Returns the log file path.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    log_path  = os.path.join(log_dir, f"alfred_{split}_{timestamp}.log")

    report = build_report_text(results, mode=mode, split=split)

    with open(log_path, "w", encoding="utf-8") as f:
        if extra_lines:
            f.write("\n".join(extra_lines) + "\n\n")
        f.write(report)

    return log_path


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import contextlib, io

    parser = argparse.ArgumentParser(
        description="Evaluate LLM intent extraction on ALFRED dataset"
    )
    parser.add_argument(
        "--alfred-path", type=str, required=True,
        help="Path to ALFRED json_2.1.0 directory (e.g. alfred/data/json_2.1.0)"
    )
    parser.add_argument(
        "--split", type=str, default="valid_seen",
        choices=["train", "valid_seen", "valid_unseen", "tests_seen", "tests_unseen"],
        help="ALFRED data split to evaluate on (default: valid_seen)"
    )
    parser.add_argument(
        "--max-cases", type=int, default=100,
        help="Number of test cases to evaluate (default: 100)"
    )
    parser.add_argument(
        "--regex-only", action="store_true",
        help="Evaluate regex fallback only (no Ollama required)"
    )
    args = parser.parse_args()

    use_ollama = not args.regex_only
    mode_label = "Ollama / Mistral + CoT" if use_ollama else "Regex fallback only"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir      = os.path.join(project_root, "test_log")

    # Determine log path before running so we can tell the user immediately
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    log_path  = os.path.join(log_dir, f"alfred_{args.split}_{timestamp}.log")

    sys.stdout.write(f"Running evaluation — output → {log_path}\n")
    sys.stdout.flush()

    # Redirect all stdout (including LLM print statements) to the log file
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        cases = load_alfred_cases(
            alfred_dir=args.alfred_path,
            split=args.split,
            max_cases=args.max_cases
        )
        if len(cases) < args.max_cases:
            print(f"[WARNING] Only {len(cases)} cases loaded — fewer than requested {args.max_cases}")

        results = evaluate(cases, use_ollama=use_ollama)

    # Build and write log: loader stats + LLM raw output + full report
    loader_output = captured.getvalue().splitlines()
    report        = build_report_text(results, mode=mode_label, split=args.split)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(loader_output) + "\n\n")
        f.write(report)

    # Single summary line to terminal
    total     = len(results)
    n_correct = sum(r["correct"] for r in results)
    sys.stdout.write(f"Done — {n_correct}/{total} correct ({n_correct/total*100:.1f}%)  |  log: {log_path}\n")
