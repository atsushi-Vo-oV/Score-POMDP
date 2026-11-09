"""One process, one task/configuration, many independent seed sessions."""
from __future__ import annotations

from pathlib import Path

from ..compare import _CAMPAIGN_PATTERN, _eligible, _selection
from ..config import ConfigError
from ..envs import make_env
from .backend import SeedBatchBackend
from .comparison import comparison_session
from .protocol import SeedStream
from .validation import validate


def run_seedbatch(config, *, campaign_id, seeds=None, task=None, resume=False,
                  segment_updates=None, max_updates=None, verbose=True):
    """Return ordinary compare summaries, in requested seed order.

    Shard numbers come from the original config's complete bulk-index matrix,
    even when a subset or a different ordering of seeds is requested.
    """
    validate(config)
    if not _CAMPAIGN_PATTERN.fullmatch(campaign_id):
        raise ConfigError("campaign_id contains unsafe characters")
    tasks = config["experiment"]["tasks"]
    if task is None:
        if len(tasks) != 1:
            raise ConfigError("seedbatch requires exactly one task; supply --task")
        task = tasks[0]
    selected = list(config["experiment"]["seeds"] if seeds is None else seeds)
    if len(set(selected)) != len(selected):
        raise ConfigError("seedbatch seeds must be unique")
    _selection(config, ["score_transformer"], [task], selected)
    if config["comparison"]["belief_gradient_modes"] != ["full"]:
        raise ConfigError("seedbatch currently requires comparison.belief_gradient_modes=[full]")
    if config["model"]["belief_gradient_mode"] != "full":
        raise ConfigError("seedbatch currently requires model.belief_gradient_mode=full")
    environment = make_env(task, {"name": task, **config.environment_for_task(task)}, seed=selected[0])
    if environment.action_spec.is_discrete:
        raise ConfigError("seedbatch currently requires continuous actions (diffusion policy)")
    if segment_updates is not None and (not resume or max_updates is not None):
        raise ConfigError("--segment-updates requires --resume and excludes --max-updates")
    for name, value in (("segment_updates", segment_updates), ("max_updates", max_updates)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ConfigError(f"{name} must be a positive integer")
    matrix = [(mode, method, task_name, seed)
              for mode in config["comparison"]["belief_gradient_modes"]
              for method in config["comparison"]["methods"]
              for task_name in tasks if _eligible(config, method, task_name)[0]
              for seed in config["experiment"]["seeds"]]
    root = Path(config["experiment"]["output_dir"]) / "campaigns" / campaign_id
    execution_config = config.with_overrides({"experiment.output_dir": str(root)})
    sessions, streams = [], []
    for seed in selected:
        shard = matrix.index(("full", "score_transformer", task, seed)) + 1
        sessions.append(comparison_session(
            execution_config, methods=["score_transformer"], tasks=[task], seeds=[seed],
            gradient_modes=["full"], campaign_id=campaign_id, bulk_index=shard,
            label=f"genkai-{campaign_id}-shard{shard:03d}", run_dir=root / f"shard{shard:03d}",
            resume=resume, segment_updates=segment_updates, max_updates=max_updates, verbose=verbose,
        ))
        streams.append(SeedStream())
    results = {}

    def advance(index, value=None):
        try:
            return streams[index].run(lambda: sessions[index].send(value))
        except StopIteration as done:
            results[index] = done.value[1][0]
            return None

    try:
        pending = [advance(i) for i in range(len(sessions))]
        active = [i for i, request in enumerate(pending) if request is not None]
        if not active:
            return [results[i] for i in range(len(sessions))]
        if any(pending[i].kind != "ready" for i in active):
            raise RuntimeError("seed session did not reach its initialization barrier")
        positions = {pending[i].payload[2:] for i in active}
        if len(positions) != 1:
            raise ConfigError("seed checkpoints are at different updates; first resume the lagging seeds alone with --seeds and --segment-updates to align them")
        models = [pending[i].payload[0] for i in active]
        optimizers = [pending[i].payload[1] for i in active]
        backend = SeedBatchBackend(models, optimizers, config["model"], config["ppo"])
        pending = {i: advance(i) for i in active}
        while any(pending.values()):
            kinds = {request.kind if request else "finished" for request in pending.values()}
            if len(kinds) != 1:
                raise RuntimeError(f"seed sessions lost synchronization: {kinds}")
            kind = next(iter(kinds))
            if kind == "belief":
                particles, scores = backend.belief([pending[i].payload for i in active])
                answers = [(particles[s], scores[s]) for s in range(len(active))]
            elif kind == "policy":
                values = backend.policy([pending[i].payload for i in active])
                answers = [tuple(value[s] for value in values) for s in range(len(active))]
            elif kind == "recondition":
                values = backend.recondition([pending[i].payload for i in active])
                answers = list(values.unbind(0))
            elif kind == "update":
                objectives = {pending[i].payload[1] for i in active}
                if len(objectives) != 1:
                    raise RuntimeError("seed prediction phases differ")
                answers = backend.update([pending[i].payload[0] for i in active],
                                         [streams[i] for i in active], next(iter(objectives)))
            else:
                raise RuntimeError(f"unknown seedbatch request: {kind}")
            pending = {i: advance(i, answer) for i, answer in zip(active, answers)}
        return [results[i] for i in range(len(sessions))]
    except Exception as exc:
        # Let each existing artifact transaction record a failed manifest and
        # traceback, including errors raised in the vectorized compute phase.
        for i, session in enumerate(sessions):
            if i not in results:
                try:
                    streams[i].run(lambda: session.throw(exc))
                except (Exception, StopIteration):
                    pass
        raise
    finally:
        for session, stream in zip(sessions, streams):
            session.close()
            stream.close()
