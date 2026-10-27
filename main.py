#!/usr/bin/env python3
"""
main.py
=======

Entry point. Example usage:

    python main.py --video my_clip.mp4
    python main.py --video my_clip.mp4 --no-hearing        # vision only
    python main.py --video my_clip.mp4 --audio-gain 2.0    # louder hearing drive
    python main.py --video my_clip.mp4 --no-summary-graph  # skip end-of-video chart
    python main.py --video my_clip.mp4 --level detailed
    python main.py --video my_clip.mp4 --level connectome     # real FlyWire, auto-download
    python main.py --video my_clip.mp4 --level connectome --max-neurons 0  # whole brain
    python main.py --video my_clip.mp4 --level connectome --connectome-path ./data/my_export
    python main.py --fetch-connectome                         # download/verify data, then exit
    python main.py --video my_clip.mp4 --no-video-panel       # 3D view only
    python main.py --make-sample-video                        # generate a synthetic test clip

Run `python main.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import sys

from simulation.config import SimLevel, SimulationConfig


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Simulated Drosophila visual + auditory system and brain, driven "
            "by a video file and its soundtrack."
        )
    )
    p.add_argument("--video", type=str, default=None, help="Path to an input video file.")
    p.add_argument(
        "--level",
        type=str,
        choices=[lvl.value for lvl in SimLevel],
        default=SimLevel.DEMO.value,
        help="Simulation scale: demo (fast, small synthetic network), detailed "
        "(larger synthetic network), connectome (real data; FlyWire is "
        "downloaded automatically unless --connectome-path is given).",
    )
    p.add_argument(
        "--connectome-path",
        type=str,
        default=None,
        help="Directory containing neurons.csv/edges.csv for --level connectome. "
        "When omitted, the real FlyWire FAFB v783 graph is downloaded and used. "
        "See data/connectome_loader.py for the CSV schema.",
    )
    p.add_argument(
        "--connectome-source",
        choices=["auto", "csv", "flywire"],
        default="auto",
        help="Real-data backend for --level connectome. 'flywire' auto-downloads "
        "the public FlyWire FAFB v783 connectome (no token, ~880 MB once); "
        "'csv' uses --connectome-path; 'auto' picks csv when a path is given, "
        "otherwise flywire. Default: auto.",
    )
    p.add_argument(
        "--connectome-cache",
        type=str,
        default=None,
        help="Directory for downloaded source files and the built-graph cache "
        "(default: data/real/flywire_v783).",
    )
    p.add_argument(
        "--max-neurons",
        type=int,
        default=None,
        help="For --connectome-source flywire: approximate size of the "
        "interactive subset (default 20000, preserving local retinotopic "
        "circuits). Use 0 for the whole 139,255-neuron brain (slower).",
    )
    p.add_argument(
        "--min-synapses",
        type=int,
        default=None,
        help="Minimum summed synapses per directed connection. Default: 1 for "
        "a subset, 5 for whole-brain mode (the published FAFB threshold).",
    )
    p.add_argument(
        "--rebuild-connectome",
        action="store_true",
        help="Ignore a cached built graph and rebuild it from the source files.",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Never download real-connectome data; fail if it is not cached.",
    )
    p.add_argument(
        "--no-motion-injection",
        action="store_true",
        help="Do not inject HRC motion maps into T4/T5 (real) or medulla "
        "(synthetic); let the connectome dynamics do all the work.",
    )
    p.add_argument(
        "--fetch-connectome",
        action="store_true",
        help="Download/cache the real FlyWire files and exit (no simulation).",
    )
    p.add_argument("--video-fps", type=float, default=15.0, help="Frames/sec fed to the simulation.")
    p.add_argument(
        "--sub-steps",
        type=int,
        default=3,
        help="Neural integration sub-steps per video frame (higher = smoother/slower dynamics).",
    )
    p.add_argument(
        "--no-hearing",
        action="store_true",
        help="Disable the auditory pathway; run vision-only (the video's "
        "soundtrack is otherwise decoded and fed to Johnston's-organ model).",
    )
    p.add_argument(
        "--audio-gain",
        type=float,
        default=1.0,
        help="Multiplier on auditory drive into the brain (default 1.0).",
    )
    p.add_argument(
        "--no-summary-graph",
        action="store_true",
        help="Do not generate the behavior-summary graph when the video ends.",
    )
    p.add_argument(
        "--summary-dir",
        type=str,
        default=None,
        help="Directory for the end-of-video summary PNG (default: data/summary).",
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
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.make_sample_video:
        from data.sample.make_sample_video import make_sample_video

        out_path = make_sample_video()
        print(f"Wrote synthetic sample video to {out_path}")
        return 0

    if args.fetch_connectome:
        from data.flywire_source import ensure_flywire_files

        files = ensure_flywire_files(
            args.connectome_cache, allow_download=not args.no_download
        )
        print(f"annotations:  {files.annotations}")
        print(f"connectivity: {files.connectivity}")
        return 0

    level = SimLevel(args.level)
    config = SimulationConfig.for_level(level, connectome_path=args.connectome_path)
    config.video_target_fps = args.video_fps
    config.steps_per_video_frame = args.sub_steps
    config.random_seed = args.seed
    config.hearing_enabled = not args.no_hearing
    config.auditory_gain = args.audio_gain
    config.summary_graph = not args.no_summary_graph
    config.summary_output_dir = args.summary_dir

    if level == SimLevel.CONNECTOME:
        source = args.connectome_source
        if source == "auto":
            source = "csv" if args.connectome_path else "flywire"
        if source == "csv" and not args.connectome_path:
            parser.error("--connectome-source csv requires --connectome-path")
        config.connectome_source = source
        config.connectome_cache_dir = args.connectome_cache
        config.max_neurons = args.max_neurons
        config.min_synapses = args.min_synapses
        config.allow_connectome_download = not args.no_download
        config.rebuild_connectome = args.rebuild_connectome
        config.motion_injection = not args.no_motion_injection

    if args.summary_only:
        from brain.network import FlyBrain

        brain = FlyBrain(config)
        g = brain.graph
        print(f"level={g.level} source={g.source}")
        print(f"neurons={g.n_neurons:,} synapses={g.adjacency.nnz:,}")
        if g.cell_type_of is not None:
            known = {str(ct) for ct in g.cell_type_of if str(ct)}
            print(f"cell types={len(known):,}")
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
