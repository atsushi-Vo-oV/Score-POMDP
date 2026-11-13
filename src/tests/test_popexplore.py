import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sb_pomdp.config import load_config as ordinary_config, ConfigError
from sb_pomdp.popexplore import PopConfig, run_population
from sb_pomdp.popexplore.actors import PopulationPolicy, OUBias, assignments
from sb_pomdp.popexplore.experiments import arm_config, ARMS
from sb_pomdp.popexplore.selection import PopulationSearch, rank_correlation, ActorAdam
from sb_pomdp.popexplore.rnd import RND, RunningMoments
from sb_pomdp.policies import DiffusionPolicy
from sb_pomdp.train import model_for_environment, train_single
from sb_pomdp.envs import make_env
from sb_pomdp.artifacts import SeedArtifacts


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny(root, arm="ppo", **overrides):
    cfg = ordinary_config("config/trysvgdlight_tagr_tf-svgd.json").with_overrides({
        "experiment.output_dir": str(root), "experiment.device": "cpu",
        "experiment.deterministic": True, "experiment.seeds": [10],
        "experiment.eval_episodes": 1, "experiment.eval_interval_updates": 2,
        "experiment.checkpoint_interval_updates": 2,
        "environment.tasks.disk_tag.horizon": 5,
        "model.num_particles": 3, "model.langevin_steps": 1,
        "model.energy_hidden": [4], "model.d_model": 8, "model.num_heads": 2,
        "model.transformer_ff_dim": 8, "model.policy_hidden": [8],
        "model.diffusion_steps": 2, "model.prediction_predictor_hidden": [8],
        "model.prediction_pretrain_steps": 0, "model.prediction_interval_updates": 0,
        "model.prediction_phase_updates": 0,
        "ppo.total_steps": 24, "ppo.num_envs": 4, "ppo.rollout_steps": 3,
        "ppo.epochs": 1, "ppo.minibatch_size": 12, "ppo.sequence_microbatch_size": 6,
    })
    p = arm_config(arm)
    if p.get("population_size", 1) > 1:
        p["population_size"] = 2
    p.update(selection_interval=1, fitness_episodes=2, rnd_hidden=8, rnd_features=4, arm=arm)
    return PopConfig({**cfg.to_dict(), "popexplore": p}).with_overrides(overrides)


def run_dir(root, campaign):
    return root / f"campaigns/{campaign}/shard001/full/score_transformer/disk_tag/seed_010"


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("arm", list(ARMS))
def test_all_arms_and_strict_checkpoint(tmp_path, arm):
    cfg = tiny(tmp_path, arm)
    result = run_population(cfg, campaign_id=arm, verbose=False)
    assert result[0]["updates"] == 2
    root = run_dir(tmp_path, arm)
    assert len(rows(root / "population.csv")) == 2 * cfg.pop["population_size"]
    assert {r["mode"] for r in rows(root / "evaluation.csv")} == {"mean_chain", "stochastic"}
    checkpoint = torch.load(root / "checkpoints/latest.pt", weights_only=False)
    # The unmodified config constructor and model loader accept this export.
    from sb_pomdp.config import ExperimentConfig
    base = ExperimentConfig(checkpoint["config"])
    env = make_env("disk_tag", base.environment_for_task("disk_tag"), seed=10)
    model = model_for_environment(env, checkpoint["config"]["model"], torch.device("cpu"))
    model.load_state_dict(checkpoint["model"], strict=True)
    metrics = rows(root / "metrics.csv")
    if cfg.pop["population_size"] > 1:
        assert int(metrics[-1]["fitness_update"]) == 2


@pytest.mark.parametrize("prediction", [False, True])
def test_off_matches_original_ppo_exactly(tmp_path, prediction):
    cfg = tiny(tmp_path)
    if prediction:
        cfg = cfg.with_overrides({"model.prediction_pretrain_steps":12,
                    "model.prediction_interval_updates":1, "model.prediction_phase_updates":1})
    run_population(cfg, campaign_id="off", verbose=False)
    path = tmp_path / "ordinary"
    path.mkdir()
    train_single(cfg.base, task="disk_tag", seed=10, artifacts=SeedArtifacts(path, "disk_tag", 10),
                 method="score_transformer", verbose=False)
    a = torch.load(run_dir(tmp_path, "off") / "checkpoints/population.pt", weights_only=False)
    b = torch.load(path / "checkpoints/latest.pt", weights_only=False)
    for name in b["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], atol=0, rtol=0)
    assert torch.equal(a["torch_rng_state"], b["torch_rng_state"])
    for name in ("metrics.csv", "episodes.csv", "evaluation.csv"):
        left, right = rows(run_dir(tmp_path, "off") / name), rows(path / name)
        assert len(left) == len(right)
        for ra, rb in zip(left, right):
            for key, value in rb.items():
                if key not in {"steps_per_second", "gpu_peak_memory_mb"}:
                    assert ra[key] == value, key


@pytest.mark.parametrize("arm", ["full", "ppo"])
def test_resume_exact(tmp_path, arm):
    cfg = tiny(tmp_path, arm, **{"ppo.total_steps": 36})
    run_population(cfg, campaign_id="whole", verbose=False)
    for _ in range(3):
        run_population(cfg, campaign_id="segments", segment_updates=1, resume=True, verbose=False)
    aroot, broot = run_dir(tmp_path, "whole"), run_dir(tmp_path, "segments")
    a = torch.load(aroot / "checkpoints/population.pt", weights_only=False)
    b = torch.load(broot / "checkpoints/population.pt", weights_only=False)
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
            for xx, yy in zip(x, y):
                equal(xx, yy)
        else:
            assert x == y or (isinstance(x, float) and np.isnan(x) and np.isnan(y))
    for key in ("model", "optimizer", "population", "torch_rng_state", "numpy_rng_state", "python_rng_state"):
        equal(a[key], b[key])
    for name in ("metrics.csv", "population.csv", "population_evaluation.csv", "evaluation.csv", "episodes.csv"):
        ar, br = rows(aroot / name), rows(broot / name)
        for row in ar + br:
            row.pop("steps_per_second", None)
        assert ar == br, name
    contents = {p: p.read_bytes() for p in broot.rglob("*") if p.is_file()}
    run_population(cfg, campaign_id="segments", resume=True, segment_updates=1, verbose=False)
    assert contents == {p: p.read_bytes() for p in contents}


def policy():
    return DiffusionPolicy(4, [-1, -1], [1, 1], [8], num_steps=2, beta_start=.01, beta_end=.1, min_std=.05)


def test_vmap_and_snapshot_match_individual():
    torch.manual_seed(3)
    p = policy()
    bank = PopulationPolicy(p, 3, 6, .02)
    c = torch.randn(6, 4)
    noise = torch.randn(6, 1, 3, 2)
    action, chain, logs = bank.call("sample", bank.episode_weights, c[:, None], noise)
    from sb_pomdp.popexplore.actors import ActorKernel
    for i in range(6):
        single = bank.export(bank.vectors()[i // 2])
        out = ActorKernel(single)("sample", c[i:i+1], noise[i])
        for batched, expected in zip((action, chain, logs), out):
            torch.testing.assert_close(batched[i], expected, atol=1e-6, rtol=1e-5)
    fresh, _ = bank.log_prob_for(c, chain[:, 0], bank.ids)
    torch.testing.assert_close(fresh, logs[:, 0], atol=1e-6, rtol=1e-5)
    old = [p.clone() for p in bank.episode_weights]
    bank.set_vectors(bank.vectors() + 1)
    initial = torch.tensor([True, False, False, False, False, True])
    bank.start_episodes(initial)
    for previous, current in zip(old, bank.episode_weights):
        assert torch.equal(previous[~initial], current[~initial])
        assert not torch.equal(previous[initial], current[initial])
    assert assignments(8, 2).tolist() == [0]*4 + [1]*4
    with pytest.raises(ValueError):
        assignments(7, 2)


@pytest.mark.parametrize("rule", ["cem", "es", "pbt", "qd", "none"])
def test_selection_rules(rule):
    cfg = tiny(Path("unused"), "full").pop
    cfg.update(update_rule=rule, elite_fraction=.25, pbt_fraction=.25, qd_parent_fraction=.5,
               sigma_min=.01, sigma=.1, mutation_std=0.0)
    vectors = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 10
    search = PopulationSearch(vectors, cfg)
    fitness = [0, 1, 2, 3]
    result, selected, parents, changed = search.update(vectors, fitness, np.eye(4))
    assert result.shape == vectors.shape and torch.isfinite(result).all()
    if rule != "none":
        assert selected.tolist() == [False, False, False, True]
    if rule == "pbt":
        assert parents[0] == 3 and torch.equal(result[0], vectors[3])
        assert not changed[1:].any()
    if rule in {"cem", "qd"}:
        assert torch.all(search.std >= .01)
        torch.testing.assert_close(search.mean, .5*vectors.mean(0)+.5*vectors[3])
    if rule == "qd":
        assert len(search.archive) == 4
    if rule == "none":
        assert torch.equal(result, vectors) and not changed.any()
    assert rank_correlation(fitness, fitness) == 1
    assert np.isnan(rank_correlation([1, 1], [2, 3]))


def test_ou_tau_and_conditional_likelihood():
    torch.manual_seed(1)
    ou = OUBias(2048, 2, .3, 12, "cpu")
    initial = torch.ones(2048, dtype=torch.bool)
    ou.step(initial)
    for _ in range(200):
        ou.step(~initial)
    rho, tau = ou.correlation()
    assert abs(tau - 12) < .5
    p = policy()
    c = torch.randn(2048, 4)
    action, chain, old = p.sample(c, deterministic=False)
    biased = torch.tanh(chain[:, -1] + ou.value)
    assert not torch.equal(biased, action)
    new, _ = p.log_prob_chain(c, chain)
    torch.testing.assert_close((new-old).exp(), torch.ones_like(old), atol=0, rtol=0)


def test_rnd_fixed_target_and_normalization():
    cfg = tiny(Path("unused")).pop
    rnd = RND(4, cfg, "cpu")
    x = torch.randn(64, 4, requires_grad=True)
    target = {k:v.clone() for k,v in rnd.target.state_dict().items()}
    before = float(rnd.error(x).mean())
    for _ in range(40):
        rnd.update_predictor(x)
    assert float(rnd.error(x).mean()) < before
    assert x.grad is None
    assert all(torch.equal(target[k], v) for k,v in rnd.target.state_dict().items())
    reward = rnd.reward(x)
    assert (reward >= 0).all() and (reward <= cfg["rnd_clip"]).all()
    moments = RunningMoments()
    values = torch.randn(100)
    moments.update(values[:30]); moments.update(values[30:])
    assert moments.mean == pytest.approx(float(values.double().mean()))
    assert moments.variance == pytest.approx(float(values.double().var(unbiased=False)))


@pytest.mark.parametrize("key,value", [("model.belief_compile", "inductor"),
    ("model.dropout", 0.1), ("model.particle_sampler", "ula"),
    ("popexplore.population_size", 3), ("popexplore.ou_tau", 0), ("popexplore.typo", 1)])
def test_unsupported_rejected(tmp_path, key, value):
    with pytest.raises((ConfigError, ValueError)):
        tiny(tmp_path).with_overrides({key:value})
    assert not (tmp_path / "campaigns").exists()


@pytest.mark.parametrize("mode", ["joint", "prediction_only"])
def test_pretrain_alternating_and_resume(tmp_path, mode):
    cfg = tiny(tmp_path, "full", **{"model.prediction_pretrain_steps": 12,
               "model.prediction_interval_updates": 1, "model.prediction_phase_updates": 1,
               "model.belief_training": mode})
    result = run_population(cfg, campaign_id="phases", segment_updates=2, resume=True, verbose=False)
    assert result[0]["updates"] == 2
    result = run_population(cfg, campaign_id="phases", segment_updates=2, resume=True, verbose=False)
    assert result[0]["updates"] == 4
    assert result[0]["status"] == "completed"
    cp = torch.load(run_dir(tmp_path,"phases")/"checkpoints/population.pt", weights_only=False)
    assert cp["population"]["ppo_updates"] == 2


def test_initial_checkpoint_freeze_and_resume_without_asset(tmp_path):
    base = tiny(tmp_path, **{"model.prediction_pretrain_steps":12,
                "model.prediction_interval_updates":1, "model.prediction_phase_updates":1})
    run_population(base, campaign_id="source", verbose=False)
    source = run_dir(tmp_path, "source") / "checkpoints/latest.pt"
    cfg = tiny(tmp_path, "full", **{"model.prediction_pretrain_steps":12,
                "model.prediction_interval_updates":1, "model.prediction_phase_updates":1,
                "model.init_from_checkpoint":str(source), "popexplore.freeze_belief":True})
    run_population(cfg, campaign_id="frozen", segment_updates=1, resume=True, verbose=False)
    before = torch.load(source, weights_only=False)["model"]
    source.rename(source.with_suffix(".hidden"))
    run_population(cfg, campaign_id="frozen", segment_updates=1, resume=True, verbose=False)
    cp = torch.load(run_dir(tmp_path,"frozen")/"checkpoints/latest.pt", weights_only=False)
    for key in before:
        if key.startswith(("belief.","encoder.","prediction_predictor.")):
            assert torch.equal(before[key], cp["model"][key]), key
    assert any(not torch.equal(before[k],cp["model"][k]) for k in before if k.startswith("value_head."))


def test_actor_data_isolation_and_inactive_adam():
    torch.manual_seed(11)
    bank = PopulationPolicy(policy(), 2, 4, .01)
    c = torch.randn(7,4)
    chain = torch.randn(7,3,2)
    ids = torch.zeros(7, dtype=torch.long)
    logs, _ = bank.log_prob_for(c, chain, ids)
    (-logs.mean()).backward()
    for p in bank.weights:
        assert torch.count_nonzero(p.grad[1]) == 0
    config = dict(learning_rate=.001, weight_decay=.1)
    optimizer = ActorAdam(bank.weights, config)
    before = bank.vectors().clone()
    optimizer.step(torch.tensor([True,False]))
    assert torch.equal(before[1], bank.vectors()[1])
    assert optimizer.steps.tolist() == [1,0]
    assert not torch.equal(before[0], bank.vectors()[0])
    optimizer.reset(torch.tensor([True,False]))
    assert optimizer.steps.tolist() == [0,0]
    assert all(torch.count_nonzero(m[0]) == 0 for m in optimizer.first)


def test_rnd_observation_and_fractional_c(tmp_path):
    cfg = tiny(tmp_path,"no_f", **{"popexplore.rnd_input":"observation", "popexplore.actor_ppo_fraction":.5})
    run_population(cfg, campaign_id="fraction", verbose=False)
    assert {r["actor_ppo_fraction"] for r in rows(run_dir(tmp_path,"fraction")/"metrics.csv")} == {"0.5"}


def test_jobs_and_budget_manifest():
    from sb_pomdp.popexplore.config import load_config
    from sb_pomdp.popexplore.runner import configured_updates
    manifest = json.loads(Path("jobs/popx/manifest.json").read_text())
    manifest = [row for row in manifest if row["arm"] in ARMS]
    assert all(r["group"] == "scratch" for r in manifest)
    assert len(manifest) == 180
    assert not list(Path("config").glob("popx_frozen_*"))
    assert not list(Path("jobs/popx").glob("*frozen*"))
    assert set(r["arm"] for r in manifest) == set(ARMS)
    assert Path("jobs/popx/scratch.queue").read_text().splitlines() == [f"TREE {r['job']}" for r in manifest]
    ordering = [(r["seed"], {"carflag":0,"mirror2":1,"tagr":2}[r["task"]]) for r in manifest]
    assert ordering == sorted(ordering)
    identities = set()
    for row in manifest:
        cfg = load_config(row["config"])
        assert cfg.base["ppo"]["num_envs"] == 64
        assert configured_updates(cfg) == row["updates"]
        assert cfg.base["experiment"]["seeds"][row["bulk_index"]-1] == row["seed"]
        ident = (row["campaign"],row["shard_index"])
        assert ident not in identities
        identities.add(ident)
        assert "SB_POMDP_ENTRY=popexplore" in Path(row["job"]).read_text()


def test_analysis_real_artifacts_and_paired_intervals(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("popx_analyze", "scripts/popx_analyze.py")
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    for arm in ("full", "ppo"):
        run_population(tiny(tmp_path,arm), campaign_id=arm, verbose=False)
    out = tmp_path / "analysis/report.md"
    result = analyzer.analyze([str(tmp_path/"campaigns")], out, plots=False)
    assert result["runs"] == 2
    report = out.read_text(encoding="utf-8")
    assert "判定不可" in report and "選抜" in report
    assert len(rows(out.with_name("report_curves.csv"))) == 4
    mean, lo, hi = analyzer.paired_interval([(10,20,1),(10,21,1),(11,20,2),(11,21,2)])
    assert mean == 1.5 and lo <= mean <= hi
    assert math_isnan(analyzer.paired_interval([(10,20,1)])[1])
    assert analyzer.paired_interval([(10,20,0),(10,21,0),(11,20,10)])[0] == 5


def math_isnan(value):
    return np.isnan(value)


def test_freeze_switch_is_reversible(tmp_path):
    cfg = tiny(tmp_path)
    frozen = cfg.with_overrides({"popexplore.freeze_belief":True})
    assert "belief." in frozen.base["model"]["freeze_modules"]
    unfrozen = frozen.with_overrides({"popexplore.freeze_belief":False})
    assert unfrozen.base["model"]["freeze_modules"] == cfg.base["model"]["freeze_modules"]


def test_export_commit_recovery_and_config_guard(tmp_path):
    cfg = tiny(tmp_path, "full")
    run_population(cfg, campaign_id="recover", verbose=False)
    path = run_dir(tmp_path,"recover")/"checkpoints/latest.pt"
    expected = torch.load(path, weights_only=False)["model"]
    path.unlink()
    run_population(cfg, campaign_id="recover", resume=True, verbose=False)
    restored = torch.load(path, weights_only=False)["model"]
    assert all(torch.equal(expected[k],restored[k]) for k in expected)
    changed = cfg.with_overrides({"popexplore.ou_std":.7})
    with pytest.raises(ValueError, match="config"):
        run_population(changed, campaign_id="recover", resume=True, verbose=False)


def test_vector_adam_matches_independent_optimizers():
    torch.manual_seed(21)
    p = torch.nn.Parameter(torch.randn(3,4))
    separate = [torch.nn.Parameter(row.detach().clone()) for row in p]
    ordinary = [torch.optim.AdamW([q],lr=.001,weight_decay=.1) for q in separate]
    stacked = ActorAdam([p],dict(learning_rate=.001,weight_decay=.1))
    for step in range(4):
        gradient = torch.randn_like(p)
        p.grad = gradient
        active = torch.tensor([True, step%2 == 0, False])
        stacked.step(active)
        for i in range(3):
            if active[i]:
                separate[i].grad = gradient[i].clone()
                ordinary[i].step()
        torch.testing.assert_close(p,torch.stack(separate),atol=2e-7,rtol=1e-6)


def test_archive_capacity_quality_and_es_ties():
    cfg = tiny(Path("unused"),"full").pop
    cfg.update(qd_capacity=2, qd_parent_fraction=0.0)
    vectors = torch.arange(20,dtype=torch.float32).reshape(4,5)
    search = PopulationSearch(vectors,cfg)
    search.update(vectors,[0,1,2,3],np.eye(4))
    assert sorted(v["fitness"] for v in search.archive.values()) == [2,3]
    search.update(vectors,[0,0,0,0],np.eye(4))
    assert sorted(v["fitness"] for v in search.archive.values()) == [2,3]
    es = PopulationSearch(vectors,{**cfg,"update_rule":"es"})
    before = es.mean.clone()
    es.update(vectors,[1,1,1,1],np.eye(4))
    assert torch.equal(before,es.mean)


def test_analytic_score_matches_autograd(tmp_path):
    """The analytic score is an execution-only switch of the shared belief; the population path only passes it through."""
    metrics = {}
    for implementation in ("autograd", "analytic"):
        root = tmp_path / implementation
        cfg = tiny(root, "full", **{"model.score_implementation": implementation})
        run_population(cfg, campaign_id="score", verbose=False)
        directory = run_dir(root, "score")
        metrics[implementation] = (rows(directory / "metrics.csv"), rows(directory / "population.csv"))
    compared = set()
    for reference, analytic in zip(*metrics.values()):
        assert len(reference) == len(analytic) > 0
        for a, b in zip(reference, analytic):
            for key in ("policy_loss", "value_loss", "fitness_return_std", "return_mean"):
                if key in a and a[key] not in ("", "nan"):
                    assert float(b[key]) == pytest.approx(float(a[key]), rel=1e-3, abs=1e-4), key
                    compared.add(key)
    assert {"policy_loss", "value_loss", "return_mean"} <= compared
