"""CLI for the isolated seed-batched path."""
import argparse
import json

from ..config import load_config
from .runner import run_seedbatch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--task")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--segment-updates", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    summaries = run_seedbatch(
        load_config(args.config, args.override), campaign_id=args.campaign_id,
        task=args.task, seeds=args.seeds, resume=args.resume,
        segment_updates=args.segment_updates, max_updates=args.max_updates, verbose=not args.quiet,
    )
    print(json.dumps(summaries, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
