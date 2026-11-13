"""Campaign layout, guarded segment resume, and one-run-per-shard execution."""
import argparse
import json
import math
from pathlib import Path

import torch

from ..artifacts import SeedArtifacts, _strict_component, _atomic_write_json
from ..config import parse_cli_overrides, resume_comparable_config
from ..envs import make_env
from ..prediction_schedule import PredictionSchedule
from .config import load_config, PopConfig
from .controller import source_hash


def configured_updates(config):
    r = config.to_dict()
    m, p = r["model"], r["ppo"]
    if config.pop["freeze_belief"]:
        m.update(prediction_pretrain_steps=0, prediction_interval_updates=0, prediction_phase_updates=0)
    batch = p["num_envs"] * p["rollout_steps"]
    return PredictionSchedule.from_config(m, rollout_batch_size=batch,
        ppo_updates=math.ceil(p["total_steps"]/batch), critic_warmup_updates=p.get("critic_warmup_updates", 0)).total_updates


def repair_inference_export(state, config, path, task, seed):
    """Recover a crash between the authoritative population commit and export."""
    from ..train import resolve_device
    from .representation import make_model
    from ..initialization import initialize_model
    from .controller import Controller
    device = resolve_device(config.base["experiment"]["device"])
    env = make_env(task, config.environment_for_task(task), seed=seed)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        model = make_model(env, config.base.to_dict(), device)
        initialize_model(model, config.base["model"], resume=True)
        artifacts = SeedArtifacts(path, task, seed, config.method, "full")
        controller = Controller(model, config.to_dict(), task, seed, artifacts, env, device, resume=True)
        model.load_state_dict(state["model"], strict=True)
        controller.restore(state["population"])
        controller.save_checkpoint(state, path / "checkpoints/latest.pt")


def run_population(config, *, campaign_id, seeds=None, bulk_index=None, shard_index=None,
                   resume=False, segment_updates=None, max_updates=None, verbose=True):
    from .session import train_single
    if not isinstance(config, PopConfig):
        config = PopConfig(config.to_dict())
    config.validate()
    if config.method == "gru":
        from .gru_session import train_single
    r = config.to_dict()
    _strict_component(campaign_id, "campaign id")
    if max_updates is not None and segment_updates is not None:
        raise ValueError("max_updates and segment_updates are mutually exclusive")
    matrix = [(task, seed) for task in r["experiment"]["tasks"] for seed in r["experiment"]["seeds"]]
    selection = [(i+1, task, seed) for i, (task, seed) in enumerate(matrix)
                 if (seeds is None or seed in seeds) and (bulk_index is None or i+1 == bulk_index)]
    if not selection or (seeds is not None and not set(seeds) <= set(r["experiment"]["seeds"])):
        raise ValueError("empty/invalid seed or bulk-index selection")
    if shard_index is not None and (len(selection) != 1 or shard_index < 1):
        raise ValueError("shard-index requires exactly one run and a positive index")
    # Reject unsupported action spaces before creating artifact directories.
    for _, task, seed in selection:
        if not make_env(task, config.environment_for_task(task), seed=seed).action_spec.is_continuous:
            raise ValueError("popexplore supports continuous actions only")
    summaries = []
    for index, task, seed in selection:
        shard = Path(r["experiment"]["output_dir"]) / "campaigns" / campaign_id / f"shard{shard_index or index:03d}"
        path = shard / "full" / config.method / task / f"seed_{seed:03d}"
        checkpoint = path / "checkpoints/population.pt"
        resume_path = None
        if path.exists() and any(path.iterdir()):
            if not resume:
                raise FileExistsError(f"run already exists: {path}; use --resume")
            if not checkpoint.is_file():
                raise ValueError(f"cannot resume without authoritative state: {checkpoint}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if resume_comparable_config(state["config"]) != resume_comparable_config(r):
                raise ValueError("resume config mismatch")
            if state["population"]["source_hash"] != source_hash():
                raise ValueError("popexplore source changed since checkpoint")
            if state["seed"] != seed or state["task"] != task:
                raise ValueError("resume identity mismatch")
            if state["update"] == configured_updates(config):
                latest = path / "checkpoints/latest.pt"
                try:
                    export = torch.load(latest, map_location="cpu", weights_only=False)
                    current = export.get("update") == state["update"] and export.get("popexplore_inference_only")
                except (OSError, RuntimeError, EOFError):
                    current = False
                if not current:
                    repair_inference_export(state, config, path, task, seed)
                summaries.append({"task": task, "seed": seed, "status": "completed", "updates": state["update"]})
                continue
            resume_path = checkpoint
        path.mkdir(parents=True, exist_ok=True)
        artifacts = SeedArtifacts(path, task, seed, config.method, "full")
        _atomic_write_json(shard / "resolved_config.json", r)
        trainer_options = {"result_dir": str(path.relative_to(shard))} if config.method == "gru" else {}
        summary = train_single(config, task=task, seed=seed, artifacts=artifacts,
                               method=config.method, resume_checkpoint=resume_path,
                               segment_updates=segment_updates, max_updates=max_updates, verbose=verbose,
                               **trainer_options)
        _atomic_write_json(shard / "summary.json", summary)
        summaries.append(summary)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--bulk-index", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--segment-updates", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)
    config = load_config(args.config, parse_cli_overrides(args.override))
    summaries = run_population(config, campaign_id=args.campaign_id, seeds=args.seeds,
        bulk_index=args.bulk_index, shard_index=args.shard_index, resume=args.resume,
        segment_updates=args.segment_updates, max_updates=args.max_updates, verbose=not args.quiet)
    print(json.dumps(summaries, ensure_ascii=False))
    return 0
