#!/usr/bin/env python3
"""
main.py
=======

Entry point. Example usage:

    python main.py --video my_clip.mp4
    python main.py --video my_clip.mp4 --level detailed
    python main.py --video my_clip.mp4 --level connectome --connectome-path ./data/real_hemibrain
    python main.py --video my_clip.mp4 --no-video-panel      # 3D view only
    python main.py --make-sample-video                       # generate a synthetic test clip

Run `python main.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import sys

from simulation.config import SimLevel, SimulationConfig


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulated Drosophila visual system + brain, driven by a video file."
    )
    p.add_argument("--video", type=str, default=None, help="Path to an input video file.")
    p.add_argument(
        "--level",
        type=str,
        choices=[lvl.value for lvl in SimLevel],
        default=SimLevel.DEMO.value,
        help="Simulation scale: demo (fast, small synthetic network), "
        "detailed (larger synthetic network), connectome (real data, requires --connectome-path).",
    )
    p.add_argument(
        "--connectome-path",
        type=str,
        default=None,
        help="Directory containing neurons.csv/edges.csv for --level connectome. "
        "See data/connectome_loader.py for the expected format and how to obtain real data.",
    )
    p.add_argument("--video-fps", type=float, default=15.0, help="Frames/sec fed to the simulation.")
    p.add_argument(
        "--sub-steps",
        type=int,
        default=3,
        help="Neural integration sub-steps per video frame (higher = smoother/slower dynamics).",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for the synthetic connectome/dynamics noise.")
    p.add_argument("--no-video-panel", action="store_true", help="Only show the 3D brain view.")
    p.add_argument(
        "--make-sample-video",
        action="store_true",
        help="Write a small synthetic demo video to data/sample/synthetic_demo.mp4 and exit.",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Build the connectome/brain and print a summary (no windows) - useful for CI/tests.",
    )
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)

    if args.make_sample_video:
        from data.sample.make_sample_video import make_sample_video

        out_path = make_sample_video()
        print(f"Wrote synthetic sample video to {out_path}")
        return 0

    level = SimLevel(args.level)
    config = SimulationConfig.for_level(level, connectome_path=args.connectome_path)
    config.video_target_fps = args.video_fps
    config.steps_per_video_frame = args.sub_steps
    config.random_seed = args.seed

    if args.summary_only:
        from brain.network import FlyBrain

        brain = FlyBrain(config)
        g = brain.graph
        print(f"level={g.level} source={g.source}")
        print(f"neurons={g.n_neurons:,} synapses={g.adjacency.nnz:,}")
        for region, sl in g.region_slices.items():
            print(f"  {region:<18s} {sl.stop - sl.start:,}")
        return 0

    if not args.video:
        print(
            "No --video given. Generating and using a synthetic demo video instead.\n"
            "Pass --video <path> to use your own footage.",
            file=sys.stderr,
        )
        from data.sample.make_sample_video import make_sample_video

        video_path = make_sample_video()
    else:
        video_path = args.video

    from ui.app import FlyBrainApp

    app = FlyBrainApp(config, video_path, show_video_panel=not args.no_video_panel)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
