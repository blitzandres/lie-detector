"""Command line interface for the text-first Blitz Engine MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

from blitz_engine.engine import BlitzEngine
from blitz_engine.reporting import dumps_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blitz", description="Blitz Engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze-text",
        help="Run the current text-only MVP against one response.",
    )
    analyze.add_argument("--baseline-file", required=True, help="Text file with one baseline sample per line.")
    analyze.add_argument("--response-file", required=True, help="Text file containing the response to analyze.")
    analyze.add_argument("--question", default="", help="Optional question prompt.")
    analyze.add_argument("--response-latency-ms", type=int, default=None, help="Optional response latency in milliseconds.")
    analyze.add_argument("--baseline-duration-s", type=float, default=120.0, help="Declared baseline duration in seconds.")
    analyze.add_argument("--use-case", default="research", help="Declared use case.")
    analyze.add_argument("--jurisdiction", default="CA-US", help="Declared jurisdiction code.")
    analyze.add_argument("--output", default="", help="Optional output JSON path.")
    analyze.add_argument(
        "--baseline-latencies-file",
        default="",
        help="Optional text file with one baseline latency per line in milliseconds.",
    )

    video = subparsers.add_parser(
        "analyze-video",
        help="Research tier: analyze a recorded response video against baseline clips "
             "(needs the [research] extra: pip install -e '.[research]').",
    )
    video.add_argument(
        "--baseline-video", action="append", required=True, dest="baseline_videos",
        help="Neutral baseline clip; repeat the flag — at least 3 clips required.",
    )
    video.add_argument("--response-video", required=True, help="The response clip to analyze.")
    video.add_argument("--question", default="", help="Optional question prompt.")
    video.add_argument("--optical-flow", action="store_true",
                       help="Add the Farneback optical-flow pass (visual.flow_agitation cue).")
    video.add_argument("--use-case", default="research", help="Declared use case.")
    video.add_argument("--jurisdiction", default="CA-US", help="Declared jurisdiction code.")
    video.add_argument("--output", default="", help="Optional output JSON path.")
    return parser


def read_lines(path: str) -> list[str]:
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    return [line for line in lines if line]


def read_optional_int_lines(path: str) -> list[int] | None:
    if not path:
        return None
    return [int(line) for line in read_lines(path)]


def run_analyze_text(args: argparse.Namespace) -> int:
    baseline_texts = read_lines(args.baseline_file)
    response_text = Path(args.response_file).read_text().strip()
    baseline_latencies = read_optional_int_lines(args.baseline_latencies_file)

    engine = BlitzEngine(modalities=["linguistic"])
    session = engine.new_session(
        baseline_texts=baseline_texts,
        baseline_latencies_ms=baseline_latencies,
        baseline_duration_s=args.baseline_duration_s,
        consent=True,
        use_case=args.use_case,
        jurisdiction=args.jurisdiction,
    )
    result = session.analyze_text(
        response_text=response_text,
        question=args.question or None,
        response_latency_ms=args.response_latency_ms,
    )

    payload = dumps_report(result)
    if args.output:
        Path(args.output).write_text(payload + "\n")
    else:
        print(payload)

    return 0


def run_analyze_video(args: argparse.Namespace) -> int:
    from modalities.visual.analyzer import VisualAnalyzer

    flow = None
    if args.optical_flow:
        from modalities.visual.flow import FarnebackFlow
        flow = FarnebackFlow()

    engine = BlitzEngine(modalities=["visual"], visual_analyzer=VisualAnalyzer(flow=flow))
    session = engine.new_session(
        baseline_video_files=args.baseline_videos,
        consent=True,
        use_case=args.use_case,
        jurisdiction=args.jurisdiction,
    )
    result = session.analyze(
        video_path=args.response_video,
        question=args.question or None,
    )

    payload = dumps_report(result)
    if args.output:
        Path(args.output).write_text(payload + "\n")
    else:
        print(payload)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze-text":
        return run_analyze_text(args)
    if args.command == "analyze-video":
        return run_analyze_video(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
