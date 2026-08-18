#!/usr/bin/env python3
"""Pareto frontier of YOLO benchmark results.

Reads the unified benchmark CSV and extracts the configurations that are
Pareto-optimal over two objectives: minimize latency (mean_ms) and maximize
accuracy. Writes a scatter + frontier plot and a CSV of the frontier points.

Usage:
    python benchmark/pareto.py [--csv benchmark/results/benchmark_results.csv]
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "results" / "benchmark_results.csv"

FORMAT_COLORS = {
    "torch": "#2ca02c",
    "onnx": "#1f77b4",
    "ncnn": "#ff7f0e",
}
MODEL_MARKERS = {}  # model name -> marker symbol, assigned on the fly


def marker_for(model: str) -> str:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    if model not in MODEL_MARKERS:
        MODEL_MARKERS[model] = markers[len(MODEL_MARKERS) % len(markers)]
    return MODEL_MARKERS[model]


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def valid(row: dict) -> bool:
    status = row.get("status", "")
    if "ERROR" in status or "SANITY" in status:
        return False
    for field in ("mean_ms", "accuracy"):
        try:
            float(row[field])
        except (KeyError, TypeError, ValueError):
            return False
    return True


def pareto_front(points: list[dict]) -> list[dict]:
    """Nondominated points: minimize mean_ms, maximize accuracy."""
    ranked = sorted(points, key=lambda p: (p["mean_ms"], -p["accuracy"]))
    front, best_acc = [], -1.0
    for p in ranked:
        if p["accuracy"] > best_acc:
            front.append(p)
            best_acc = p["accuracy"]
    return front


def label(row: dict) -> str:
    quant = "" if row["quantize"] == "fp32" else f" {row['quantize']}"
    rebuilt = " (net/frame)" if "net rebuilt per frame" in row.get("status", "") else ""
    return f"{row['model']}|{row['format']}{quant}@{row['imgsz']}{rebuilt}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pareto frontier of YOLO benchmark results")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Unified benchmark CSV")
    parser.add_argument("--out-png", default=None, help="Output plot path")
    parser.add_argument("--out-csv", default=None, help="Frontier CSV path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    all_rows = load_rows(csv_path)
    fields = list(all_rows[0].keys()) if all_rows else []
    points = [p for p in all_rows if valid(p)]
    if not points:
        raise SystemExit("No valid rows (numeric mean_ms + accuracy, no ERROR/SANITY).")

    for p in points:
        p["mean_ms"] = float(p["mean_ms"])
        p["accuracy"] = float(p["accuracy"])

    front = pareto_front(points)
    front_set = {id(p) for p in front}

    out_png = Path(args.out_png) if args.out_png else csv_path.parent / "pareto_frontier.png"
    out_csv = Path(args.out_csv) if args.out_csv else csv_path.parent / "pareto_frontier.csv"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    for fmt in sorted({p["format"] for p in points}):
        pts = [p for p in points if p["format"] == fmt]
        for p in pts:
            alpha = 1.0 if id(p) in front_set else 0.45
            ax.scatter(
                p["mean_ms"], p["accuracy"],
                color=FORMAT_COLORS.get(fmt, "#333"),
                marker=marker_for(p["model"]),
                s=60, alpha=alpha, edgecolors="k", linewidths=0.5,
            )

    fx = [p["mean_ms"] for p in front]
    fy = [p["accuracy"] for p in front]
    ax.step(fx, fy, where="post", color="crimson", lw=2, alpha=0.8, label="Pareto frontier")
    for p in front:
        ax.annotate(
            label(p), (p["mean_ms"], p["accuracy"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )

    handles = [
        plt.Line2D([], [], color=FORMAT_COLORS.get(f, "#333"), marker="o", ls="",
                   label=f"{f} (color)")
        for f in sorted(FORMAT_COLORS)
    ]
    handles += [
        plt.Line2D([], [], color="#888", marker=mk, ls="", label=f"{name} (marker)")
        for name, mk in MODEL_MARKERS.items()
    ]
    handles += [plt.Line2D([], [], color="crimson", lw=2, label="Pareto frontier")]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel("Latency (mean_ms) — lower is better")
    ax.set_ylabel("Accuracy (0-1) — higher is better")
    ax.set_title(f"YOLO benchmark Pareto frontier ({len(points)} configs, {len(front)} optimal)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Plot -> {out_png}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(front)
    print(f"Frontier ({len(front)} rows) -> {out_csv}")

    print(f"\nPareto-optimal configurations ({len(front)}):")
    for p in front:
        rebuilt = " [net/frame]" if "net rebuilt per frame" in p.get("status", "") else ""
        print(f"  {label(p):<32} mean={p['mean_ms']:8.2f} ms  acc={p['accuracy']:.3f}{rebuilt}")


if __name__ == "__main__":
    main()