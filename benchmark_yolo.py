#!/usr/bin/env python3
"""Benchmark YOLO models across formats, quantizations and image sizes.

Compares torch / onnx (fp32, fp16) / ncnn (fp32, fp16) on frames extracted
from an MP4 video and writes per-config latency stats to a CSV.

NCNN corrupts its allocator when switching input sizes within one process, so
each (ncnn config, imgsz) pair is benchmarked in a fresh subprocess.

Usage:
    python benchmark_yolo.py --model yolo26n.pt --video video.mp4
"""

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2

from ultralytics import YOLO
from ultralytics import __version__ as UL_VERSION

EXPORT_DIR = Path("exports")
OUTPUT_CSV = "benchmark_results.csv"

# (format, quantize_label, export_kwargs)
CONFIGS = [
    ("torch", "fp32", None),
    ("onnx", "fp32", {"dynamic": True, "simplify": True}),
    ("onnx", "fp16", {"dynamic": True, "simplify": True, "quantize": 16}),
    ("ncnn", "fp32", {"quantize": None}),
    ("ncnn", "fp16", {"quantize": 16}),
]

IMGSZ = [320, 480, 640, 800]
WARMUP = 2
EXPORT_IMGSZ = 640
PERSON_CLASS = 0
EXPECTED_PERSONS = 4
CONF = 0.4
SANITY_THRESHOLD = 3.0


def extract_frames(video: Path, n: int) -> list:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = n
    indices = [round(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    frames, pos = [], -1
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames could be read from {video}")
    return frames


def ncnn_model_dir(cfg_dir: Path) -> Path | None:
    for p in cfg_dir.rglob("*.ncnn.param"):
        return p.parent
    return None


def cfg_dir_for(model: Path, fmt: str, quantize: str) -> Path:
    return EXPORT_DIR / model.stem / f"{fmt}_{quantize}"


def source_fingerprint(model: Path, export_kwargs: dict | None) -> dict:
    sha = hashlib.sha256()
    with open(model, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return {
        "source_sha256": sha.hexdigest(),
        "imgsz": EXPORT_IMGSZ,
        "export_args": export_kwargs or {},
        "ultralytics": UL_VERSION,
    }


def marker_matches(cfg_dir: Path, expected: dict) -> bool:
    marker = cfg_dir / "marker.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text()) == expected
    except (json.JSONDecodeError, OSError):
        return False


def ensure_export(
    model: Path, fmt: str, quantize: str, export_kwargs: dict | None
) -> tuple[str, float]:
    if fmt == "torch":
        return str(model), 0.0
    cfg_dir = cfg_dir_for(model, fmt, quantize)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    expected = source_fingerprint(model, export_kwargs)

    if fmt == "onnx":
        artifact = cfg_dir / "model.onnx"
        if artifact.exists() and marker_matches(cfg_dir, expected):
            return str(artifact), 0.0
    elif fmt == "ncnn":
        existing = ncnn_model_dir(cfg_dir)
        if existing is not None and marker_matches(cfg_dir, expected):
            return str(existing), 0.0
        artifact = None
    else:
        raise ValueError(fmt)

    # Stale or missing artifact: wipe the config dir and re-export.
    shutil.rmtree(cfg_dir, ignore_errors=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    src = cfg_dir / "model.pt"
    shutil.copy2(model, src)
    m = YOLO(src)
    t0 = time.perf_counter()
    m.export(format=fmt, imgsz=EXPORT_IMGSZ, **(export_kwargs or {}))
    elapsed = time.perf_counter() - t0
    (cfg_dir / "marker.json").write_text(json.dumps(expected, indent=2))

    if fmt == "onnx":
        if not artifact.exists():
            raise RuntimeError(f"Export produced no artifact at {artifact}")
        return str(artifact), elapsed
    found = ncnn_model_dir(cfg_dir)
    if found is None:
        raise RuntimeError(f"Export produced no NCNN model dir under {cfg_dir}")
    return str(found), elapsed


def benchmark_one(
    model: Path,
    fmt: str,
    quantize: str,
    export_kwargs: dict | None,
    frames: list,
    imgsz: int,
    fresh_per_frame: bool = False,
) -> dict:
    artifact, export_s = ensure_export(model, fmt, quantize, export_kwargs)
    m = YOLO(artifact, task="detect")
    note = ""
    for _ in range(WARMUP):
        if fresh_per_frame:
            m = YOLO(artifact, task="detect")
        m.predict(frames[0], imgsz=imgsz, verbose=False)
    times, persons = [], []
    for frame in frames:
        if fresh_per_frame:
            m = YOLO(artifact, task="detect")
        t0 = time.perf_counter()
        results = m.predict(frame, imgsz=imgsz, verbose=False, conf=CONF)
        times.append((time.perf_counter() - t0) * 1000)
        persons.append(int((results[0].boxes.cls == PERSON_CLASS).sum()))
    mean = statistics.fmean(times)
    if fresh_per_frame:
        note += " (net rebuilt per frame)"
    return {
        "format": fmt,
        "quantize": quantize,
        "imgsz": imgsz,
        "mean_ms": round(mean, 3),
        "median_ms": round(statistics.median(times), 3),
        "std_ms": round(statistics.pstdev(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "fps": round(1000 / mean, 2),
        "avg_persons": round(statistics.fmean(persons), 3),
        "accuracy": round(
            statistics.fmean(
                min(p, EXPECTED_PERSONS) / EXPECTED_PERSONS for p in persons
            ),
            3,
        ),
        "status": "ok" + note,
    }


def error_row(fmt: str, quantize: str, imgsz: int, err: Exception) -> dict:
    return {
        "format": fmt,
        "quantize": quantize,
        "imgsz": imgsz,
        "mean_ms": "",
        "median_ms": "",
        "std_ms": "",
        "min_ms": "",
        "max_ms": "",
        "fps": "",
        "avg_persons": "",
        "accuracy": "",
        "status": f"ERROR: {err}",
    }


def write_csv(rows: list, out: Path) -> None:
    fields = [
        "format",
        "quantize",
        "imgsz",
        "mean_ms",
        "median_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "fps",
        "avg_persons",
        "accuracy",
        "status",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows -> {out}")


def run_worker(args: argparse.Namespace) -> dict:
    model, video = Path(args.model), Path(args.video)
    config = next(c for c in CONFIGS if c[0] == args.fmt and c[1] == args.quant)
    frames = extract_frames(video, args.frames)
    try:
        fresh = args.fmt == "ncnn" and args.imgsz[0] != EXPORT_IMGSZ
        return benchmark_one(
            model,
            config[0],
            config[1],
            config[2],
            frames,
            args.imgsz[0],
            fresh_per_frame=fresh,
        )
    except Exception as e:  # noqa: BLE001
        return error_row(config[0], config[1], args.imgsz[0], e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO export formats/quantizations/image sizes"
    )
    parser.add_argument(
        "--model", default="yolo26n.pt", help="Source .pt model"
    )
    parser.add_argument(
        "--video", default="video.avi", help="MP4 video for test frames"
    )
    parser.add_argument(
        "--frames", type=int, default=60, help="Number of test frames"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=IMGSZ,
        help="Image sizes (multiples of 32)",
    )
    parser.add_argument("--out", default=OUTPUT_CSV, help="CSV output path")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete this model's exports and re-export from scratch",
    )
    parser.add_argument(
        "--fmt",
        choices={c[0] for c in CONFIGS},
        help="Worker mode: format to benchmark",
    )
    parser.add_argument(
        "--quant",
        choices={c[1] for c in CONFIGS},
        help="Worker mode: quantize to benchmark",
    )
    args = parser.parse_args()

    if args.fmt or args.quant:
        row = run_worker(args)
        print(json.dumps(row))
        return

    model, video = Path(args.model), Path(args.video)
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.clean:
        model_dir = EXPORT_DIR / model.stem
        if model_dir.exists():
            shutil.rmtree(model_dir)
            print(f"Cleaned {model_dir}")

    frames = extract_frames(video, args.frames)
    print(f"Extracted {len(frames)} frames from {video}")

    rows = []
    for fmt, quantize, export_kwargs in CONFIGS:
        for imgsz in args.imgsz:
            if fmt == "ncnn":
                cmd = [
                    sys.executable,
                    str(Path(__file__)),
                    "--model",
                    str(model),
                    "--video",
                    str(video),
                    "--frames",
                    str(args.frames),
                    "--fmt",
                    fmt,
                    "--quant",
                    quantize,
                    "--imgsz",
                    str(imgsz),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                line = (
                    proc.stdout.strip().splitlines()[-1]
                    if proc.stdout.strip()
                    else ""
                )
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, IndexError):
                    detail = f" (exit {proc.returncode})"
                    tail = proc.stderr.strip().splitlines()
                    if tail:
                        detail += f": {tail[-1][:200]}"
                    row = error_row(
                        fmt,
                        quantize,
                        imgsz,
                        RuntimeError(f"worker crashed{detail}"),
                    )
            else:
                try:
                    row = benchmark_one(
                        model, fmt, quantize, export_kwargs, frames, imgsz
                    )
                except Exception as e:  # noqa: BLE001
                    row = error_row(fmt, quantize, imgsz, e)
            rows.append(row)
            print(
                f"  [{fmt} {quantize}] imgsz={imgsz}: "
                f"{row.get('mean_ms', '-')} ms ({row.get('fps', '-')} fps) "
                f"persons={row.get('avg_persons', '-')} acc={row.get('accuracy', '-')} "
                f"{row.get('status', '')}"
            )

    refs = {
        r["imgsz"]: r["avg_persons"]
        for r in rows
        if r["format"] == "torch" and isinstance(r["avg_persons"], (int, float))
    }
    for r in rows:
        if r["format"] == "torch" or not isinstance(r["avg_persons"], (int, float)):
            continue
        ref = refs.get(r["imgsz"])
        if ref is None:
            continue
        delta = abs(r["avg_persons"] - ref)
        if delta > SANITY_THRESHOLD:
            r["status"] = f"{r['status']} | SANITY vs torch Δ={delta:.1f} persons".strip()

    write_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()
