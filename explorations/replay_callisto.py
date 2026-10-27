"""
Replay recorded autofocus runs with one or more versions of astrafocus and compare the results.

Each run is a reference exposure at the start position followed by one sweep, as taken by Astra
with `AnalyticResponseAutofocuser`. Runs are split where consecutive frames are more than
`--gap` seconds apart.

By default, the focus positions are the ones astrafocus commanded: `integer_linspace(start -
half_range, start + half_range, n)`, in the order of the recorded sweep. On Callisto, the
`FOCUSPOS` header differs from these by up to about 60 steps, and the first sweep frame was taken
while the focuser was still moving. `--positions header` uses the header values instead.

For every version and star-size measure, this script

- measures the focus in every sweep frame, with the reference frame used to select the stars,
- replays the full autofocus run (`ObservationBasedDeviceSimulator`), and
- writes the results to `<out>/<label>.json` and a plot per run to `<out>/run<i>.png`.

Each version runs in its own Python process, so different source trees do not mix.

Usage
-----
    git worktree add ../astrafocus-pr2 pr-2   # e.g. after `git fetch origin pull/2/head:pr-2`
    python explorations/replay_callisto.py ../20261022 ../replay_out \\
        --src main=src --src pr2=../astrafocus-pr2/src --true-focus 7010

    # Use the 5th frame of each run (index into the run, 0 = reference) as the reference image:
    python explorations/replay_callisto.py ../20261022 ../replay_out --src pr2=... --ref-index 5
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

MEASURES = ("HFRStarFocusMeasure", "GaussianStarFocusMeasure")
TIME_PATTERN = re.compile(r"_(\d{8}_\d{6}\.\d+)\.fits$")


def load_runs(data_dir: Path, gap: float):
    """Return the runs as lists of (path, timestamp) sorted by time, split at gaps > `gap` seconds."""
    frames = []
    for path in data_dir.glob("*.fits"):
        match = TIME_PATTERN.search(path.name)
        if match:
            frames.append((path, datetime.strptime(match.group(1), "%Y%m%d_%H%M%S.%f")))
    frames.sort(key=lambda frame: frame[1])

    runs = [[frames[0]]]
    for previous, frame in zip(frames, frames[1:]):
        if (frame[1] - previous[1]).total_seconds() > gap:
            runs.append([])
        runs[-1].append(frame)
    return runs


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        message = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            message += f" [{type(record.exc_info[1]).__name__}: {record.exc_info[1]}]"
        self.messages.append(message)


def worker(args):
    """Measure all runs with the astrafocus found in `args.src` and write `<out>/<label>.json`."""
    sys.path.insert(0, str(Path(args.src).resolve()))
    from astropy.io import fits

    import astrafocus
    import astrafocus.star_size_focus_measure_operators as ssfm
    from astrafocus.autofocuser import AnalyticResponseAutofocuser
    from astrafocus.interface.simulation import ObservationBasedDeviceSimulator

    assert Path(astrafocus.__file__).resolve().is_relative_to(Path(args.src).resolve()), astrafocus.__file__

    collect = _Collect()
    logging.getLogger("astrafocus").addHandler(collect)
    logging.getLogger("astrafocus").propagate = False

    results = {"label": args.label, "src": str(Path(args.src).resolve()), "runs": []}
    for i_run, run in enumerate(load_runs(Path(args.data), args.gap)):
        images, focus_pos = [], []
        for path, _ in run:
            with fits.open(path) as hdul:
                images.append(hdul[0].data.astype(np.float32))
                focus_pos.append(float(hdul[0].header["FOCUSPOS"]))
                exposure_time = float(hdul[0].header["EXPTIME"])
        ref_image, sweep_images, header_pos = images[args.ref_index], images[1:], focus_pos[1:]
        if args.positions == "header":
            sweep_pos = header_pos
        else:
            start = int(round(focus_pos[0]))
            sweep_pos = np.round(
                np.linspace(start - args.half_range, start + args.half_range, len(header_pos))
            ).astype(int)
            if header_pos[-1] < header_pos[0]:
                sweep_pos = sweep_pos[::-1]
            sweep_pos = sweep_pos.tolist()

        run_result = {
            "run": i_run + 1,
            "start": run[0][1].isoformat(),
            "files": [path.name for path, _ in run],
            "ref_position": focus_pos[0],
            "ref_index": args.ref_index,
            "positions": args.positions,
            "sweep_pos": sweep_pos,
            "header_pos": header_pos,
            "measures": {},
        }
        for name in MEASURES:
            cls = getattr(ssfm, name)
            entry = {"values": [], "errors": [], "seconds": [], "star_sizes": []}
            try:
                op = cls(ref_image=ref_image)
                stars = op.star_finder.selected_stars
                entry["n_stars"] = len(stars)
                entry["star_xy"] = np.column_stack([stars["x_centroid"], stars["y_centroid"]]).tolist()
            except Exception as error:
                entry["star_finder_error"] = f"{type(error).__name__}: {error}"
                op = None

            for image in sweep_images:
                t0 = time.perf_counter()
                value, error = np.nan, None
                if op is not None:
                    try:
                        value = float(op.measure_focus(image))
                    except Exception as e:
                        error = f"{type(e).__name__}: {e}"
                entry["seconds"].append(time.perf_counter() - t0)
                entry["values"].append(value)
                entry["errors"].append(error)
                sizes = getattr(op, "last_star_sizes", None)
                entry["star_sizes"].append(None if sizes is None else np.asarray(sizes).tolist())

            # Only on versions with the per-star filter of PR #2.
            if op is not None and hasattr(op, "combine_star_sizes"):
                measured = [i for i, v in enumerate(entry["values"]) if np.isfinite(v)]
                combined = np.full(len(sweep_pos), np.nan)
                if measured:
                    combined[measured] = op.combine_star_sizes(
                        np.vstack([entry["star_sizes"][i] for i in measured]),
                        np.asarray(sweep_pos)[measured],
                    )
                entry["combined"] = combined.tolist()

            # Replay the whole autofocus run: the reference exposure first, then the sweep. The
            # simulator needs sorted positions to move to a best focus between them.
            collect.messages.clear()
            replay_pos = np.round([focus_pos[0], *sweep_pos]).astype(int)
            order = np.argsort(replay_pos, kind="stable")
            replay_images = [ref_image, *sweep_images]
            simulator = ObservationBasedDeviceSimulator(
                image_data=[replay_images[i] for i in order],
                headers=[None] * len(images),
                focus_pos=replay_pos[order],
                current_position=int(replay_pos[0]),
            )
            autofocuser = AnalyticResponseAutofocuser(
                autofocus_device_manager=simulator,
                exposure_time=exposure_time,
                focus_measure_operator=cls,
                n_steps=(len(sweep_pos),),
                n_exposures=(1,),
                search_range=(int(round(min(sweep_pos))), int(round(max(sweep_pos)))),
                decrease_search_range=False,
            )
            # Sweep exactly the recorded positions, in the recorded order.
            autofocuser.integer_linspace = lambda *_, positions=sweep_pos: np.round(positions).astype(int)
            success = autofocuser.run()
            entry["autofocus"] = {
                "success": bool(success),
                "best_focus": None
                if autofocuser.best_focus_position is None
                else float(autofocuser.best_focus_position),
                "final_position": float(simulator.focuser.position),
                "warnings": list(collect.messages),
            }
            run_result["measures"][name] = entry
            print(f"[{args.label}] run {i_run + 1} {name}: {entry['autofocus']}", flush=True)
        results["runs"].append(run_result)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(results, indent=1, default=float))


def plot(out: Path, labels, true_focus):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {label: json.loads((out / f"{label}.json").read_text()) for label in labels}
    n_runs = len(next(iter(results.values()))["runs"])
    summary = []
    for i_run in range(n_runs):
        fig, axes = plt.subplots(len(MEASURES), len(labels), figsize=(5 * len(labels), 7), squeeze=False)
        for col, label in enumerate(labels):
            run = results[label]["runs"][i_run]
            x = np.asarray(run["sweep_pos"])
            for row, name in enumerate(MEASURES):
                ax, entry = axes[row, col], run["measures"][name]
                ax.plot(x, entry["values"], "o", color="k", label="measure_focus")
                if "combined" in entry:
                    ax.plot(x, entry["combined"], "x", color="C0", label="combine_star_sizes")
                af = entry["autofocus"]
                if af["best_focus"] is not None:
                    ax.axvline(af["best_focus"], color="C3", ls="--", label="autofocus best focus")
                if true_focus is not None:
                    ax.axvline(true_focus, color="C2", ls=":", label="reference focus")
                status = "ok" if af["success"] else "FAILED"
                ax.set_title(f"{label}: {name.removesuffix('StarFocusMeasure')} ({status})", fontsize=10)
                ax.set_xlabel("focuser position")
                ax.legend(fontsize=7)
                n_ok = int(np.isfinite(entry["values"]).sum())
                summary.append(
                    (i_run + 1, label, name, n_ok, len(x), entry.get("n_stars"), status, af["best_focus"])
                )
        fig.suptitle(
            f"Run {i_run + 1} ({run['start']}, ref index {run['ref_index']}, {run['positions']} positions)"
        )
        fig.tight_layout()
        fig.savefig(out / f"run{i_run + 1}.png", dpi=110)
        plt.close(fig)

    header = f"{'run':>3} {'version':<8} {'measure':<26} {'measured':>9} {'stars':>5} {'result':<7}"
    print(f"\n{header} best focus")
    for run, label, name, n_ok, n, n_stars, status, best in summary:
        print(f"{run:>3} {label:<8} {name:<26} {n_ok:>4}/{n:<4} {n_stars!s:>5} {status:<7} {best}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data", help="directory with the FITS frames")
    parser.add_argument("out", help="output directory")
    parser.add_argument("--src", action="append", default=[], help="label=path/to/src, repeatable")
    parser.add_argument("--gap", type=float, default=120.0, help="seconds between frames that split runs")
    parser.add_argument("--ref-index", type=int, default=0, help="frame of each run used as reference")
    parser.add_argument("--positions", choices=("commanded", "header"), default="commanded")
    parser.add_argument("--half-range", type=int, default=1000, help="commanded sweep: start +- this")
    parser.add_argument("--true-focus", type=float, default=None, help="draw this focus position")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        args.src = args.src[0]
        return worker(args)

    sources = dict(item.split("=", 1) for item in args.src) or {"main": "src"}
    for label, src in sources.items():
        subprocess.run(
            [sys.executable, "-I", __file__, args.data, args.out, "--worker", "--label", label,
             "--src", src, "--gap", str(args.gap), "--ref-index", str(args.ref_index),
             "--positions", args.positions, "--half-range", str(args.half_range)],
            check=True,
        )  # fmt: skip
    plot(Path(args.out), list(sources), args.true_focus)


if __name__ == "__main__":
    main()
