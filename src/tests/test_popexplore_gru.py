import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sb_pomdp.artifacts import SeedArtifacts
from sb_pomdp.config import ConfigError, ExperimentConfig
from sb_pomdp.envs import make_env
from sb_pomdp.popexplore import run_population
from sb_pomdp.popexplore.config import PopConfig
from sb_pomdp.train_baselines import train_baseline_single, _make_model
from test_popexplore import tiny as light_tiny, rows, threads


def tiny(root, arm="gru_ppo", **overrides):
    config = light_tiny(root, arm.removeprefix("gru_")).to_dict()
    config["comparison"].update(methods=["gru"], gru_encoder_hidden=[8], gru_hidden_dim=6)
    config["popexplore"]["arm"] = arm
    return PopConfig(config).with_overrides(overrides)


def run_dir(root, campaign):
    return root / f"campaigns/{campaign}/shard001/full/gru/disk_tag/seed_010"


def equal(x, y):
    if torch.is_tensor(x):
        assert torch.equal(x, y)
    elif isinstance(x, np.ndarray):
        np.testing.assert_array_equal(x, y)
    elif isinstance(x, dict):
        assert x.keys() == y.keys()
        for key in x:
            equal(x[key], y[key])
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y)
        for a, b in zip(x, y):
            equal(a, b)
    else:
        assert x == y or isinstance(x, float) and np.isnan(x) and np.isnan(y)


@pytest.mark.parametrize("prediction", [False, True])
@pytest.mark.parametrize("mode", ["full", "tbptt_1"])
def test_gru_off_matches_original_exactly(tmp_path, prediction, mode):
    cfg = tiny(tmp_path, **{"model.belief_gradient_mode": mode, "comparison.belief_gradient_modes": [mode],
                           "ppo.bootstrap_on_truncation": True})
    if prediction:
        cfg = cfg.with_overrides({"model.prediction_pretrain_steps": 12,
                                 "model.prediction_interval_updates": 1, "model.prediction_phase_updates": 1})
    run_population(cfg, campaign_id="off", verbose=False)
    path = tmp_path / "ordinary"
    path.mkdir()
    train_baseline_single(cfg.base, method="gru", task="disk_tag", seed=10,
        artifacts=SeedArtifacts(path, "disk_tag", 10, "gru", "full"), result_dir="ordinary", verbose=False)
    a = torch.load(run_dir(tmp_path, "off") / "checkpoints/population.pt", weights_only=False)
    b = torch.load(path / "checkpoints/latest.pt", weights_only=False)
    for key in ("model", "optimizer", "torch_rng_state", "numpy_rng_state", "python_rng_state"):
        equal(a[key], b[key])
    for name in ("metrics.csv", "episodes.csv", "evaluation.csv"):
        left, right = rows(run_dir(tmp_path, "off") / name), rows(path / name)
        assert len(left) == len(right)
        for ra, rb in zip(left, right):
            for key, value in rb.items():
                if key not in {"steps_per_second", "gpu_peak_memory_mb"}:
                    assert ra[key] == value, key


@pytest.mark.parametrize("arm", ["gru_ppo", "gru_full"])
@pytest.mark.parametrize("prediction", [False, True])
def test_gru_resume_and_strict_load(tmp_path, arm, prediction):
    cfg = tiny(tmp_path, arm, **{"ppo.total_steps": 36, "ppo.bootstrap_on_truncation": True})
    if prediction:
        cfg = cfg.with_overrides({"model.prediction_pretrain_steps": 12,
                                 "model.prediction_interval_updates": 1, "model.prediction_phase_updates": 1})
    run_population(cfg, campaign_id="whole", verbose=False)
    from sb_pomdp.popexplore.runner import configured_updates
    for _ in range(configured_updates(cfg)):
        run_population(cfg, campaign_id="split", segment_updates=1, resume=True, verbose=False)
    aroot, broot = run_dir(tmp_path, "whole"), run_dir(tmp_path, "split")
    a = torch.load(aroot / "checkpoints/population.pt", weights_only=False)
    b = torch.load(broot / "checkpoints/population.pt", weights_only=False)
    for key in ("model", "optimizer", "population", "torch_rng_state", "numpy_rng_state", "python_rng_state"):
        equal(a[key], b[key])
    for name in ("metrics.csv", "population.csv", "population_evaluation.csv", "evaluation.csv", "episodes.csv"):
        left, right = rows(aroot / name), rows(broot / name)
        for row in left + right:
            row.pop("steps_per_second", None)
        assert left == right, name
    checkpoint = torch.load(aroot / "checkpoints/latest.pt", weights_only=False)
    base = ExperimentConfig(checkpoint["config"])
    env = make_env("disk_tag", base.environment_for_task("disk_tag"), seed=10)
    model = _make_model(env, "gru", base["model"], base["comparison"], torch.device("cpu"))
    model.load_state_dict(checkpoint["model"], strict=True)
    assert "trainer_state" not in checkpoint and checkpoint["method"] == "gru"
    if arm == "gru_full":
        equal(model.heads.value_head.state_dict(), {k.removeprefix("optimizer_value_head."):v
              for k,v in a["model"].items() if k.startswith("optimizer_value_head.")})
    before = {p:p.read_bytes() for p in broot.rglob("*") if p.is_file()}
    run_population(cfg, campaign_id="split", resume=True, verbose=False)
    assert before == {p:p.read_bytes() for p in before}


@pytest.mark.parametrize("arm", ["no_f", "no_a2", "no_c", "no_d", "no_qd", "cem", "es", "pbt", "a2", "d"])
def test_gru_component_combinations(tmp_path, arm):
    cfg = tiny(tmp_path, "gru_" + arm)
    run_population(cfg, campaign_id=arm, verbose=False)
    result = rows(run_dir(tmp_path, arm) / "metrics.csv")
    assert len(result) == 2
    assert all(np.isfinite(float(r["policy_loss"])) for r in result)


@pytest.mark.parametrize("key,value", [
    ("comparison.methods", ["gru", "score_transformer"]), ("comparison.methods", ["rnn"]),
    ("comparison.methods", ["particle_filter"]), ("model.continuous_policy_kind", "gaussian"),
    ("model.critic_kind", "action"), ("model.policy_observation_input", True),
    ("model.prediction_head", "belief_mixture"), ("ppo.self_imitation.enabled", True),
    ("ppo.weight_decay", .01), ("ppo.critic_warmup_updates", 1),
    ("popexplore.population_size", 3), ("model.freeze_modules", ["heads.diffusion_policy."]),
])
def test_gru_rejects_unsupported(tmp_path, key, value):
    with pytest.raises(ConfigError):
        tiny(tmp_path, "gru_full", **{key:value})


def test_gru_ignores_particle_runtime_configuration(tmp_path):
    cfg = tiny(tmp_path).with_overrides({"model.particle_sampler":"ula", "model.score_parameterization":"energy",
        "model.score_implementation":"autograd", "model.belief_cuda_graphs":True,
        "model.belief_compile":"off"})
    run_population(cfg, campaign_id="ignored", verbose=False)
    ordinary = tiny(tmp_path)
    run_population(ordinary, campaign_id="ordinary", verbose=False)
    a = torch.load(run_dir(tmp_path, "ignored") / "checkpoints/population.pt", weights_only=False)
    b = torch.load(run_dir(tmp_path, "ordinary") / "checkpoints/population.pt", weights_only=False)
    equal(a["model"], b["model"])
    equal(a["torch_rng_state"], b["torch_rng_state"])


@pytest.mark.parametrize("training", ["joint", "prediction_only", "frozen"])
def test_gru_prediction_training_and_freezing(tmp_path, training):
    cfg = tiny(tmp_path, "gru_full", **{"model.prediction_pretrain_steps":12,
        "model.prediction_interval_updates":1, "model.prediction_phase_updates":1,
        "ppo.max_grad_norm":.1})
    cfg = cfg.with_overrides({"popexplore.freeze_belief":True} if training == "frozen" else {"model.belief_training":training})
    run_population(cfg, campaign_id="whole", verbose=False)
    from sb_pomdp.popexplore.runner import configured_updates
    for _ in range(configured_updates(cfg)):
        run_population(cfg, campaign_id="split", resume=True, segment_updates=1, verbose=False)
    a = torch.load(run_dir(tmp_path, "whole") / "checkpoints/population.pt", weights_only=False)
    b = torch.load(run_dir(tmp_path, "split") / "checkpoints/population.pt", weights_only=False)
    for key in ("model", "optimizer", "population", "torch_rng_state"):
        equal(a[key], b[key])
    if training == "frozen":
        from sb_pomdp.train import seed_everything
        seed_everything(10, True)
        env = make_env("disk_tag", cfg.environment_for_task("disk_tag"), seed=10)
        initial = _make_model(env, "gru", cfg.base["model"], cfg.base["comparison"], torch.device("cpu"))
        for k, value in initial.state_dict().items():
            if not k.startswith("heads."):
                assert torch.equal(value, a["model"][k]), k


def make_controller(root, **overrides):
    from sb_pomdp.popexplore.controller import Controller
    cfg = tiny(root, "gru_full", **overrides)
    env = make_env("disk_tag", cfg.environment_for_task("disk_tag"), seed=10)
    model = _make_model(env, "gru", cfg.base["model"], cfg.base["comparison"], torch.device("cpu"))
    c = Controller(model, cfg.to_dict(), "disk_tag", 10, SeedArtifacts(root, "disk_tag", 10, "gru", "full"), env, torch.device("cpu"))
    object.__setattr__(model, "_popx", c)
    c.begin_update(1, "ppo")
    return c


def test_gru_f_rnd_bc_and_batch_evaluation(tmp_path):
    from sb_pomdp.popexplore.evaluation import evaluate_population, evaluate_batched
    c = make_controller(tmp_path)
    inputs = torch.randn(4, c.model.observation_dim)
    features, _ = c.model.step(inputs, c.model.initial_hidden(4), torch.ones(4, dtype=torch.bool))
    sink = [(features, features)]
    action, chain, _ = c.bank.sample(features, deterministic=False)
    changed = c.perturb_action(action, chain, torch.ones(4, dtype=torch.bool), "ppo")
    assert torch.equal(action[c.ids == 0], changed[c.ids == 0])
    reward = c.reward(torch.ones(4), inputs[:, :-2], sink, "ppo")
    assert torch.equal(reward[c.ids == 0], torch.ones(2))
    assert (reward[c.ids != 0] > 1).all()
    assert torch.equal(c.rnd_inputs[0], features.detach()) and not c.rnd_inputs[0].requires_grad
    c.model.zero_grad()
    c.values(features, c.ids)[c.ids == 0].sum().backward()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in c.model.optimizer_value_head.parameters())
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in c.model.heads.value_head.parameters())
    c.save_teacher(sink, changed)
    assert not c.teacher_conditions[0].requires_grad and torch.equal(c.teacher_actions[0], changed)
    before = c.vectors().clone()
    critic = {k:v.clone() for k,v in c.model.optimizer_value_head.state_dict().items()}
    assert np.isfinite(c.adopt(1))
    assert not torch.equal(c.vectors()[0], before[0]) and torch.equal(c.vectors()[1:], before[1:])
    equal(critic, c.model.optimizer_value_head.state_dict())
    original_heads = c.model.heads
    inferred = c.inference_model()
    assert inferred.heads is not original_heads and c.model.heads.diffusion_policy is c.bank
    assert inferred.heads.value_head is c.model.optimizer_value_head
    rng = torch.get_rng_state().clone()
    snapshot = [w.clone() for w in c.bank.episode_weights]
    result = evaluate_population(c.model, **c.eval_args(3, 42, False))
    assert result["returns"].shape == (2, 3) and (result["lengths"] == 5).all()
    evaluate_batched(inferred, **c.eval_args(3, 42, False))
    assert torch.equal(rng, torch.get_rng_state())
    equal(snapshot, c.bank.episode_weights)


@pytest.mark.parametrize("mode", ["copy", "bc"])
def test_gru_resume_through_adoption(tmp_path, monkeypatch, mode):
    from sb_pomdp.popexplore import controller as module
    real_evaluate = module.evaluate_population
    def fitness(model, **kwargs):
        result = real_evaluate(model, **kwargs)
        if kwargs["seed"] >= 2_000_000:
            result["returns"][0] = 0
            result["returns"][1:] = 1
        return result
    monkeypatch.setattr(module, "evaluate_population", fitness)
    cfg = tiny(tmp_path, "gru_full", **{"ppo.total_steps":48, "popexplore.selection_interval":2,
        "popexplore.optimizer_agent.window":2, "popexplore.optimizer_agent.adopt":mode})
    run_population(cfg, campaign_id="whole", verbose=False)
    for _ in range(4):
        run_population(cfg, campaign_id="split", resume=True, segment_updates=1, verbose=False)
    a = torch.load(run_dir(tmp_path,"whole") / "checkpoints/population.pt", weights_only=False)
    b = torch.load(run_dir(tmp_path,"split") / "checkpoints/population.pt", weights_only=False)
    assert a["population"]["adoption_gate"]["count"] == 2
    for key in ("model", "optimizer", "population", "torch_rng_state"):
        equal(a[key], b[key])
    for name in ("metrics.csv", "population.csv", "population_evaluation.csv"):
        left, right = rows(run_dir(tmp_path, "whole") / name), rows(run_dir(tmp_path, "split") / name)
        for row in left + right:
            row.pop("steps_per_second", None)
        assert left == right
    logs = rows(run_dir(tmp_path, "whole") / "metrics.csv")
    assert np.isnan(float(logs[0]["optimizer_return_before"]))
    assert np.isfinite(float(logs[1]["optimizer_return_before"]))
    assert np.isfinite(float(logs[2]["optimizer_return_pre_ppo"]))


def test_gru_completed_export_recovery(tmp_path):
    cfg = tiny(tmp_path, "gru_full")
    run_population(cfg, campaign_id="repair", verbose=False)
    path = run_dir(tmp_path, "repair") / "checkpoints/latest.pt"
    before = torch.load(path, weights_only=False)
    path.unlink()
    run_population(cfg, campaign_id="repair", resume=True, verbose=False)
    equal(before["model"], torch.load(path, weights_only=False)["model"])


def load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gru_generated_assets_and_idempotence(tmp_path, monkeypatch):
    import shutil
    from sb_pomdp.popexplore.experiments import ARMS, GRU_ARMS, arm_config
    from sb_pomdp.popexplore.config import load_config
    from sb_pomdp.popexplore.runner import configured_updates
    manifest = json.loads(Path("jobs/popx/manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 210
    assert all(r["arm"] in ARMS for r in manifest[:180])
    gru = manifest[180:]
    assert len(gru) == 30 and {r["arm"] for r in gru} == set(GRU_ARMS)
    assert len({r["config"] for r in gru}) == 6
    assert Path("jobs/popx/gru.queue").read_text().splitlines() == [f"TREE {r['job']}" for r in gru]
    assert [(r["seed"], r["task"], r["arm"]) for r in gru] == [
        (s, t, a) for s in range(10,15) for t in ("carflag", "mirror2", "tagr") for a in GRU_ARMS]
    for row in gru:
        cfg = load_config(row["config"])
        source = json.loads(Path(f"config/trysvgd_{row['task']}_gru.json").read_text(encoding="utf-8-sig"))
        assert cfg.base["comparison"] == ExperimentConfig(source)["comparison"]
        assert cfg.base["model"]["policy_hidden"] == source["model"]["policy_hidden"]
        assert cfg.base["ppo"]["num_envs"] == 64 and cfg.base["ppo"]["rollout_steps"] == 128
        assert cfg.base["ppo"]["minibatch_size"] == cfg.base["ppo"]["sequence_microbatch_size"] == 2048
        assert cfg.base["model"]["prediction_pretrain_steps"] == 98 * 64 * 128
        assert configured_updates(cfg) == row["updates"] == (1293 if row["task"] == "tagr" else 333)
        assert row["campaign"] == f"popx-{row['task']}-{row['arm'].replace('_', '-')}-v1"
        assert row["seed"] == cfg.base["experiment"]["seeds"][row["bulk_index"] - 1]
        for key, value in arm_config(row["arm"]).items():
            if key != "optimizer_agent":
                assert cfg.pop[key] == value
        script = Path(row["job"]).read_bytes()
        assert b"\r" not in script and f"SB_POMDP_CONFIG={row['config']}".encode() in script
        assert f"SB_POMDP_SEGMENT_UPDATES:-{row['updates']}".encode() in script
    # Regenerate a private copy: previous light bytes and the appended assets survive.
    files = list(Path("config").glob("popx_scratch_*.json")) + list(Path("jobs/popx").glob("popx_scratch_*.sh"))
    files += [Path(f"jobs/popx/{name}") for name in ("manifest.json", "scratch.queue", "gru.queue")]
    files += [Path(f"config/{prefix}{task}_{suffix}.json") for task in ("carflag", "mirror2", "tagr")
              for prefix, suffix in (("trysvgdlight_", "tf-svgd"), ("trysvgd_", "gru"))]
    for path in files:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    before = {str(path): (tmp_path / path).read_bytes() for path in files}
    generator = load_script("popx_generate")
    monkeypatch.setattr(generator, "__file__", str(tmp_path / "scripts/popx_generate.py"))
    generator.main()
    assert before == {str(path): (tmp_path / path).read_bytes() for path in files}


def test_gru_analysis_pairs_and_pretrain_only(tmp_path):
    analyzer = load_script("popx_analyze")
    curves = []
    # Unequal progress: seed10 pairs at u2, seed11 at u4; extra unmatched seed ignored.
    for arm, offset in (("ppo", 4), ("full", 8), ("gru_ppo", 1), ("gru_full", 3)):
        for seed in (10, 11):
            for update in ([2] if seed == 10 and arm.startswith("gru_") else [2, 4]):
                curves.append(dict(group="scratch", task="disk_tag", mode="stochastic", arm=arm,
                    checkpoint_seed=-1, seed=seed, update=update, inference=offset + seed + update))
    effects = analyzer.effects(curves)
    wanted = {("gru_full", "gru_ppo"):2, ("full", "gru_full"):5, ("ppo", "gru_ppo"):3}
    for r in effects:
        key = r["arm"], r["reference"]
        if key in wanted:
            assert r["pairs"] == 2 and r["updates"] == "2,4"
            assert r["difference"] == r["ci_low"] == r["ci_high"] == wanted[key]
    assert set(wanted) <= {(r["arm"], r["reference"]) for r in effects}
    cfg = tiny(tmp_path, "gru_full", **{"model.prediction_pretrain_steps":24,
                                        "experiment.eval_interval_updates":100})
    run_population(cfg, campaign_id="partial", segment_updates=1, verbose=False)
    out = tmp_path / "analysis.md"
    result = analyzer.analyze([str(tmp_path / "campaigns")], out, plots=False)
    text = out.read_text(encoding="utf-8")
    assert result["curve_rows"] == 0 and "## GRU 比較" in text and "評価未到達: 1" in text
    assert "gru_full の F 診断" in text


def test_gru_bench_small(tmp_path):
    from types import SimpleNamespace
    bench = load_script("popx_bench")
    result = bench.benchmark(SimpleNamespace(output_root=str(tmp_path), warmup=1, repeats=1,
        cadence="stress", cpu_small=True, fitness_from_rollout=False, tasks=["carflag"],
        arms=["gru_ppo", "gru_full"]))
    assert len(result["results"]) == 2
    assert all(row["mean_seconds"] > 0 for row in result["results"])
