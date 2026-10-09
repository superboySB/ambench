# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Export compact, traceable GIF/MP4 previews from videos or LeRobot images.

The specs JSON is a list of dictionaries. Each requires ``name``, ``source_report``
and ``outcome_path`` (a list of dictionary keys or array indices), plus exactly one
of ``source_video``, ``source_parquet``, ``session_root_path``, ``image_dir`` or
``images_glob``. ``session_root_path`` selects a session directory from the report
and reads its single LeRobot data shard. Non-video sources require ``source_fps``.
Optional keys include ``title``, ``outcome_labels``, ``image_column``, ``font_path``
and ``notes``. ``metadata`` stores fixed experiment conditions;
``metadata_paths`` reads conditions from the source report. Only declared
condition fields are accepted. Relative paths use ``--source-root``.

Example outcome selector: ["rollouts", 0, "termination_reason"]. Labels are taken
from that report value; an unknown value fails instead of guessing an outcome.
``--write-mp4`` also writes full-color H.264 previews using every sampled frame.
This is a CPU-only export and does not initialize Isaac Sim.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import math
import re
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

OUTCOME_LABELS = {"success": "SUCCESS", "timeout": "TIMEOUT", "passed": "SMOKE"}
CONDITION_FIELDS = {
    "experiment_id",
    "category",
    "trial_id",
    "camera",
    "task_id",
    "env_seed",
    "inference_seed",
    "inference_device",
    "training_seed",
    "training_steps",
    "training_reference",
    "episode_index",
    "steps",
    "action_semantics",
    "episode_length_s",
    "simulation_duration_s",
    "initial_observation",
    "source_run_date",
    "source_time_basis",
    "geometry_randomized",
    "attempt_selection",
}


def sha256(path: Path) -> str:
    """Hash a source or preview without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frames(spec: dict, source_root: Path, frame_limit: int) -> tuple[list[Image.Image], dict]:
    """Read sampled RGB frames while retaining their original indices and times."""
    source_keys = ("source_video", "source_parquet", "session_root_path", "image_dir", "images_glob")
    selected_keys = [key for key in source_keys if key in spec]
    if len(selected_keys) != 1:
        raise ValueError(f"{spec['name']}: specify exactly one frame source")
    source_key = selected_keys[0]
    frames = []
    provenance = {"source_type": source_key}
    if source_key == "session_root_path":
        session_value = json.loads((source_root / spec["source_report"]).read_text())
        for key in spec["session_root_path"]:
            session_value = session_value[key]
        session_root = source_root / session_value
        if not session_root.is_dir() and "outputs" in session_root.parts:
            anchor = session_root.parts.index("outputs")
            session_root = source_root / Path(*session_root.parts[anchor:])
        shards = sorted((session_root / "lerobot" / "data").rglob("*.parquet"))
        if len(shards) != 1:
            raise ValueError(f"Expected one LeRobot data shard for this preview, found {len(shards)}: {session_root}")
        source = shards[0]
        provenance.update({"session_root_path": spec["session_root_path"], "session_root_value": session_value})
    else:
        source = source_root / spec[source_key]
    provenance["source"] = str(source.relative_to(source_root)) if source.is_relative_to(source_root) else str(source)

    if source_key == "source_video":
        capture = cv2.VideoCapture(str(source))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if count < 2 or not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"Video has invalid frame count or FPS: {source}")
            indices = np.unique(np.linspace(0, count - 1, min(frame_limit, count), dtype=int)).tolist()
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                okay, frame = capture.read()
                if not okay:
                    raise ValueError(f"Could not decode video frame {index}: {source}")
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            capture.release()
        timestamps = [index / fps for index in indices]
        provenance["source_sha256"] = sha256(source)
    elif source_key in {"source_parquet", "session_root_path"}:
        column = spec.get("image_column", "observation.images.ee_camera")
        table = pq.read_table(source, columns=[column, "timestamp"])
        count = table.num_rows
        fps = float(spec["source_fps"])
        if count < 2 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"{spec['name']}: at least two rows and positive source_fps are required")
        indices = np.unique(np.linspace(0, count - 1, min(frame_limit, count), dtype=int)).tolist()
        selected = table.take(indices).to_pydict()
        timestamps = [float(value) for value in selected["timestamp"]]
        image_hashes = []
        for value in selected[column]:
            encoded = value.get("bytes")
            if encoded is None:
                raise ValueError(f"Parquet image has no embedded bytes: {source}")
            with Image.open(io.BytesIO(encoded)) as image:
                frames.append(image.convert("RGB"))
            image_hashes.append(hashlib.sha256(encoded).hexdigest())
        provenance.update({
            "source_sha256": sha256(source),
            "image_column": column,
            "selected_image_sha256": image_hashes,
        })
        last_time = float(table.column("timestamp")[count - 1].as_py()) if count else 0.0
        first_time = float(table.column("timestamp")[0].as_py()) if count else 0.0
        provenance["source_duration_s"] = last_time - first_time
    else:
        files = list(source.iterdir()) if source_key == "image_dir" else [Path(path) for path in glob.glob(str(source))]
        files = [path for path in files if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        files.sort(key=lambda path: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))])
        count = len(files)
        fps = float(spec["source_fps"])
        if count < 2 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"{spec['name']}: at least two images and positive source_fps are required")
        indices = np.unique(np.linspace(0, count - 1, min(frame_limit, count), dtype=int)).tolist()
        selected_files = []
        full_digest = hashlib.sha256()
        for path in files:
            relative_path = str(path.relative_to(source_root)) if path.is_relative_to(source_root) else str(path)
            full_digest.update(f"{relative_path}\0{sha256(path)}\n".encode())
        for index in indices:
            path = files[index]
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))
            relative_path = str(path.relative_to(source_root)) if path.is_relative_to(source_root) else str(path)
            selected_files.append({"path": relative_path, "sha256": sha256(path)})
        timestamps = [index / fps for index in indices]
        provenance.update({"source_images_sha256": full_digest.hexdigest(), "selected_files": selected_files})

    if count < 2 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{spec['name']}: at least two frames and positive source_fps are required")
    provenance.update({
        "source_frame_count": count,
        "source_fps": fps,
        "source_duration_s": provenance.get("source_duration_s", (count - 1) / fps),
        "source_frame_indices": indices,
        "source_timestamps_s": timestamps,
    })
    return frames, provenance


def render_frames(
    frames: list[Image.Image],
    timestamps: list[float],
    title: str,
    outcome: str,
    width: int,
    font_path: Path | None,
    even_dimensions: bool = False,
) -> list[Image.Image]:
    """Render full-color previews with labels above the unoccluded camera image."""
    if even_dimensions:
        width += width % 2
    font_size = max(9, round(width / 25))
    font = ImageFont.truetype(str(font_path), font_size) if font_path else ImageFont.load_default(size=font_size)
    previews = []
    for index, frame in enumerate(frames):
        height = max(1, round(frame.height * width / frame.width))
        label_height = font_size * 2 + 10
        total_height = height + label_height
        if even_dimensions:
            total_height += total_height % 2
        preview = Image.new("RGB", (width, total_height), (16, 24, 36))
        preview.paste(frame.resize((width, height), Image.Resampling.LANCZOS), (0, label_height))
        draw = ImageDraw.Draw(preview)
        draw.text((5, 2), title, font=font, fill="white")
        color = (110, 245, 140) if outcome == "SUCCESS" else (255, 220, 110)
        draw.text(
            (5, font_size + 5),
            f"{outcome} | t={timestamps[index]:.2f}/{timestamps[-1]:.2f}s",
            font=font,
            fill=color,
        )
        previews.append(preview)
    return previews


def encode_preview(
    frames: list[Image.Image],
    timestamps: list[float],
    title: str,
    outcome: str,
    width: int,
    colors: int,
    frame_count: int,
    duration_ms: int,
    hold_ms: int,
    font_path: Path | None,
) -> tuple[bytes, list[int], list[int]]:
    """Encode a full-episode sampling with a labeled, held final frame."""
    indices = np.unique(np.linspace(0, len(frames) - 1, min(frame_count, len(frames)), dtype=int)).tolist()
    # Keep the original end time in the progress label when GIF sampling changes.
    previews = render_frames(
        [frames[index] for index in indices],
        [timestamps[index] for index in indices],
        title,
        outcome,
        width,
        font_path,
    )
    # A shared palette keeps stationary surfaces stable between frames, allowing
    # GIF delta compression to retain more movement within the preview budget.
    tile_width = min(96, width)
    tile_height = max(1, round(previews[0].height * tile_width / width))
    columns = min(8, len(previews))
    palette_sheet = Image.new("RGB", (tile_width * columns, tile_height * math.ceil(len(previews) / columns)))
    for index, preview in enumerate(previews):
        palette_sheet.paste(
            preview.resize((tile_width, tile_height), Image.Resampling.LANCZOS),
            ((index % columns) * tile_width, (index // columns) * tile_height),
        )
    palette = palette_sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    previews = [preview.quantize(palette=palette, dither=Image.Dither.NONE) for preview in previews]
    per_frame_ms = max(10, round((duration_ms - hold_ms) / len(previews) / 10) * 10)
    durations = [per_frame_ms] * len(previews)
    durations[-1] += hold_ms
    output = io.BytesIO()
    previews[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=previews[1:],
        duration=durations,
        loop=0,
        optimize=True,
        palette=palette.getpalette(),
        disposal=1,
    )
    return output.getvalue(), indices, durations


def encode_mp4(
    frames: list[Image.Image],
    timestamps: list[float],
    title: str,
    outcome: str,
    width: int,
    fps: int,
    duration_s: float,
    hold_s: float,
    max_bytes: int,
    font_path: Path | None,
) -> tuple[bytes, list[int], int]:
    """Encode all sampled frames in color, increasing CRF to meet the byte limit."""
    previews = render_frames(frames, timestamps, title, outcome, width, font_path, even_dimensions=True)
    active_count = max(len(frames), round((duration_s - hold_s) * fps))
    indices = np.round(np.linspace(0, len(frames) - 1, active_count)).astype(int).tolist()
    indices.extend([len(frames) - 1] * round(hold_s * fps))
    # FFmpeg faststart needs a real path to reopen while moving the MP4 header.
    with tempfile.TemporaryDirectory(prefix="ambench-animation-") as temporary_dir:
        temporary_video = Path(temporary_dir) / "preview.mp4"
        for crf in (18, 22, 26, 30, 34, 38, 42, 46):
            with av.open(str(temporary_video), mode="w", options={"movflags": "+faststart"}) as container:
                stream = container.add_stream("libx264", rate=fps)
                stream.width, stream.height = previews[0].size
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": str(crf), "preset": "medium", "threads": "2"}
                for frame_number, index in enumerate(indices):
                    frame = av.VideoFrame.from_ndarray(np.asarray(previews[index]), format="rgb24")
                    frame.pts = frame_number
                    frame.time_base = Fraction(1, fps)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            encoded = temporary_video.read_bytes()
            if len(encoded) <= max_bytes:
                return encoded, indices, crf
    raise ValueError(f"MP4 remains {len(encoded)} bytes after CRF 46, exceeding {max_bytes}")


def main() -> int:
    """Export each requested preview below its byte limit or fail explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs-json", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--duration-s", type=float, default=7.0)
    parser.add_argument("--final-hold-s", type=float, default=1.0)
    parser.add_argument("--max-bytes", type=int, default=450 * 1024)
    parser.add_argument("--write-mp4", action="store_true", help="Also export full-color, fully decoded H.264 previews")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    if args.width < 120 or args.fps < 1 or args.max_bytes < 1024:
        parser.error("Width must be >= 120, FPS >= 1 and max-bytes >= 1024")
    if not 0 <= args.final_hold_s < args.duration_s:
        parser.error("Require 0 <= final-hold-s < duration-s")
    specs = json.loads(args.specs_json.read_text())
    if not isinstance(specs, list) or not specs:
        parser.error("Specs must be a nonempty JSON list")
    source_root = args.source_root.expanduser().resolve()
    names = [spec["name"] for spec in specs]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", name) for name in names):
        parser.error("Names must be unique lowercase file names containing only a-z, 0-9, _, - and .")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in specs:
        report_path = source_root / spec["source_report"]
        report = json.loads(report_path.read_text())
        outcome_value = report
        for key in spec["outcome_path"]:
            outcome_value = outcome_value[key]
        labels = spec.get("outcome_labels", OUTCOME_LABELS)
        label_key = str(outcome_value).lower()
        if label_key not in labels:
            raise ValueError(f"{spec['name']}: no outcome label for report value {outcome_value!r}")
        outcome = labels[label_key]
        conditions = dict(spec.get("metadata", {}))
        for name, selector in spec.get("metadata_paths", {}).items():
            value = report
            for key in selector:
                value = value[key]
            conditions[name] = value
        unknown_fields = set(conditions) - CONDITION_FIELDS
        if unknown_fields:
            raise ValueError(f"{spec['name']}: unsupported condition metadata: {sorted(unknown_fields)}")
        frame_limit = max(2, round((args.duration_s - args.final_hold_s) * args.fps))
        frames, provenance = load_frames(spec, source_root, frame_limit)
        font_path = source_root / spec["font_path"] if spec.get("font_path") else None
        durations = []
        chosen = []
        candidates = [(args.width, colors, frame_limit) for colors in (128, 64, 32, 16)]
        widths = {max(120, round(args.width * scale)) for scale in (0.875, 0.75, 0.625)}
        widths.update(min(args.width, width) for width in (180, 160, 128, 120))
        widths = sorted(widths, reverse=True)
        candidates.extend((width, colors, frame_limit) for width in widths for colors in (32, 16, 8))
        candidates.extend((widths[-1], 8, max(2, round(frame_limit * scale))) for scale in (0.8, 0.6, 0.5, 0.4, 0.25))
        for width, colors, frame_count in candidates:
            encoded, chosen, durations = encode_preview(
                frames,
                provenance["source_timestamps_s"],
                spec.get("title", spec["name"]),
                outcome,
                width,
                colors,
                frame_count,
                round(args.duration_s * 1000),
                round(args.final_hold_s * 1000),
                font_path,
            )
            if len(encoded) <= args.max_bytes:
                break
        else:
            raise ValueError(f"{spec['name']}: GIF remains {len(encoded)} bytes, exceeding {args.max_bytes}")
        output = args.output_dir / f"{spec['name']}.gif"
        output.write_bytes(encoded)
        original_indices = provenance["source_frame_indices"]
        original_times = provenance["source_timestamps_s"]
        mp4_metadata = {}
        if args.write_mp4:
            mp4_bytes, mp4_indices, crf = encode_mp4(
                frames,
                original_times,
                spec.get("title", spec["name"]),
                outcome,
                args.width,
                args.fps,
                args.duration_s,
                args.final_hold_s,
                args.max_bytes,
                font_path,
            )
            mp4_output = args.output_dir / f"{spec['name']}.mp4"
            mp4_output.write_bytes(mp4_bytes)
            with av.open(str(mp4_output)) as decoded:
                stream = decoded.streams.video[0]
                mp4_width, mp4_height = stream.width, stream.height
                mp4_fps = float(stream.average_rate)
                mp4_duration = float(stream.duration * stream.time_base)
                mp4_codec = stream.codec_context.name
                mp4_count = 0
                mp4_pixel_formats = set()
                for mp4_count, frame in enumerate(decoded.decode(stream), start=1):
                    if frame.width != mp4_width or frame.height != mp4_height:
                        raise ValueError(f"Decoded MP4 frame dimensions changed: {mp4_output}")
                    if not math.isclose(frame.time, (mp4_count - 1) / args.fps, abs_tol=1e-6):
                        raise ValueError(f"Decoded MP4 presentation times differ from the export: {mp4_output}")
                    mp4_pixel_formats.add(frame.format.name)
                    frame.to_ndarray(format="rgb24")
            if mp4_codec != "h264" or mp4_pixel_formats != {"yuv420p"}:
                raise ValueError(f"Decoded MP4 codec or pixel format differs from the export: {mp4_output}")
            if mp4_count != len(mp4_indices) or not math.isclose(mp4_fps, args.fps):
                raise ValueError(f"Decoded MP4 frame count or FPS differs from the export: {mp4_output}")
            if not math.isclose(mp4_duration, mp4_count / mp4_fps, abs_tol=1 / mp4_fps):
                raise ValueError(f"Decoded MP4 duration differs from the export: {mp4_output}")
            mp4_metadata.update({
                "mp4": mp4_output.name,
                "mp4_sha256": sha256(mp4_output),
                "mp4_bytes": mp4_output.stat().st_size,
                "mp4_width": mp4_width,
                "mp4_height": mp4_height,
                "mp4_frame_count": mp4_count,
                "mp4_fps": mp4_fps,
                "mp4_duration_s": mp4_duration,
                "mp4_codec": mp4_codec,
                "mp4_pixel_format": next(iter(mp4_pixel_formats)),
                "mp4_crf": crf,
                "mp4_source_frame_indices": [original_indices[index] for index in mp4_indices],
                "mp4_source_timestamps_s": [original_times[index] for index in mp4_indices],
                "mp4_final_hold_ms": round(args.final_hold_s * args.fps) / args.fps * 1000,
                "mp4_full_decode_passed": True,
            })
            for key in ("selected_image_sha256", "selected_files"):
                if key in provenance:
                    mp4_metadata[f"mp4_{key}"] = [provenance[key][index] for index in mp4_indices]
        provenance["source_frame_indices"] = [original_indices[index] for index in chosen]
        provenance["source_timestamps_s"] = [original_times[index] for index in chosen]
        for key in ("selected_image_sha256", "selected_files"):
            if key in provenance:
                provenance[key] = [provenance[key][index] for index in chosen]
        with Image.open(output) as decoded:
            decoded_duration = 0
            for index in range(decoded.n_frames):
                decoded.seek(index)
                decoded.load()
                decoded_duration += decoded.info.get("duration", 0)
            gif_frame_count = decoded.n_frames
        manifest.append({
            **provenance,
            **mp4_metadata,
            **conditions,
            "name": spec["name"],
            "title": spec.get("title", spec["name"]),
            "outcome": outcome,
            "outcome_value": outcome_value,
            "outcome_path": spec["outcome_path"],
            "outcome_labels": labels,
            "source_report": spec["source_report"],
            "source_report_sha256": sha256(report_path),
            "notes": spec.get("notes", ""),
            "gif": output.name,
            "gif_sha256": sha256(output),
            "gif_bytes": output.stat().st_size,
            "gif_width": width,
            "gif_palette_colors": colors,
            "gif_frame_count": gif_frame_count,
            "gif_duration_ms": decoded_duration,
            "sample_durations_ms": durations,
            "final_hold_ms": round(args.final_hold_s * 1000),
        })
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"{spec['name']}: {outcome}, {gif_frame_count} frames, {output.stat().st_size} bytes")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Exported {len(manifest)} recorded animations to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
