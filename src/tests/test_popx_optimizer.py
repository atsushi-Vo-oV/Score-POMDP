from types import SimpleNamespace
from pathlib import Path
import copy
import json

import numpy as np
import pytest
import torch

from test_popexplore import tiny, run_dir, rows, threads
from sb_pomdp.artifacts import SeedArtifacts
from sb_pomdp.envs import make_env
from sb_pomdp.train import model_for_environment
from sb_pomdp.popexplore import PopConfig, run_population
from sb_pomdp.popexplore.controller import Controller
from sb_pomdp.popexplore.optimizer_agent import AdoptionGate, student_quantile, noise_prediction_loss, imitate
from sb_pomdp.popexplore.config import ConfigError, OPTIMIZER_DEFAULTS


def controller(root, **overrides):
    cfg = tiny(root, "full", **{"popexplore.population_size":4,
                "popexplore.optimizer_agent.adopt":"none", **overrides})
    env = make_env("disk_tag", cfg.environment_for_task("disk_tag"), seed=10)
    model = model_for_environment(env, cfg.base["model"], torch.device("cpu"))
    root.mkdir(exist_ok=True)
    c = Controller(model, cfg.to_dict(), "disk_tag", 10, SeedArtifacts(root, "disk_tag", 10), env, torch.device("cpu"))
    object.__setattr__(model, "_popx", c)
    c.begin_update(1, "ppo")
    return c


def test_f_masks_initialization_value_and_export(tmp_path):
    c = controller(tmp_path)
    assert torch.equal(c.vectors()[0], c.original)
    assert not torch.equal(c.vectors()[1], c.original)
    inputs = torch.randn(4, 8)
    action, chain, _ = c.bank.sample(inputs, deterministic=False)
    changed = c.perturb_action(action, chain, torch.ones(4, dtype=torch.bool), "ppo")
    assert torch.equal(action[:1], changed[:1])
    assert not torch.equal(action[1:], changed[1:])
    assert torch.count_nonzero(c.ou.value[0]) == 0
    c.perturb_action(action, chain, torch.zeros(4, dtype=torch.bool), "ppo")
    assert c.ou.pairs == 3 * 2  # Only explorers, never F, enter the OU diagnostic.
    ext = torch.randn(4)
    reward = c.reward(ext, torch.randn(4, 5), [(inputs, inputs)], "ppo")
    assert torch.equal(reward[:1], ext[:1])
    assert (reward[1:] > ext[1:]).all()
    c.model.zero_grad()
    values = c.values(inputs, c.ids)
    values[0].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in c.model.optimizer_value_head.parameters())
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in c.model.value_head.parameters())
    export = c.inference_model()
    actual = torch.cat([p.detach().flatten() for p in export.diffusion_policy.parameters()])
    assert torch.equal(actual, c.vectors()[0])
    assert "optimizer_value_head" not in export._modules
    assert export.value_head is c.model.optimizer_value_head


@pytest.mark.parametrize("rule", ["cem", "es", "qd", "pbt"])
def test_f_excluded_from_selection(tmp_path, monkeypatch, rule):
    c = controller(tmp_path, **{"popexplore.update_rule":rule})
    # Deliberately make F the best: it must still be absent from the search pool.
    returns = np.array([[99.,99.],[1.,1.],[2.,2.],[3.,3.]])
    monkeypatch.setattr("sb_pomdp.popexplore.controller.evaluate_population", lambda *a, **k:dict(returns=returns, descriptors=np.eye(4)))
    monkeypatch.setattr("sb_pomdp.popexplore.controller.evaluate", lambda *a, **k:({"mean_return":99.}, {}))
    initial = c.vectors().clone()
    old_mean = c.search.mean.clone()
    c.external = [torch.ones(4)]
    c.finish_update(SimpleNamespace(returns=torch.ones(2,4), old_values=torch.zeros(2,4)), [])
    assert torch.equal(c.vectors()[0], initial[0])
    result = rows(tmp_path / "population.csv")
    assert result[0]["selected"] == result[0]["replaced"] == "0"
    assert result[0]["parent"] == "0"
    if rule in {"cem", "qd"}:
        torch.testing.assert_close(c.search.mean, .5*old_mean + .5*initial[3])
    assert all(v["fitness"] < 99 for v in c.search.archive.values())


def test_paired_gate_margin_variance_lineage():
    assert student_quantile(.95, 4) == pytest.approx(2.131846786, rel=1e-7)
    f = {**OPTIMIZER_DEFAULTS, "window":5, "margin":.1}
    gate = AdoptionGate(f, 3)
    for u in range(1,5):
        assert not gate.observe(u, [u, u+.2, u+2*(-1)**u])["eligible"]
    decision = gate.observe(5, [5,5.2,3])
    assert decision["eligible"] and decision["candidate"] == 1 and decision["lower"] > 0
    gate.reset([False, True, True])
    assert gate.observe(6, [0,.2,.2])["pairs"] == 1
    gate.observe(7, [0, np.nan, .2])
    assert not gate.history[1]
    gate = AdoptionGate(f, 2)
    for u in range(5):
        result = gate.observe(u, [0,.1])
    assert not result["eligible"]  # strict margin, not >=


@pytest.mark.parametrize("mode", ["copy", "bc", "none"])
def test_adoption_actions_and_optimizer_reset(tmp_path, mode):
    c = controller(tmp_path, **{"popexplore.optimizer_agent.adopt":mode, "popexplore.optimizer_agent.bc_steps":4})
    c.teacher_conditions = [torch.randn(4,8,requires_grad=True) for _ in range(3)]
    c.teacher_actions = [torch.zeros(4,2) for _ in range(3)]
    before = c.vectors().clone()
    value = copy.deepcopy(c.model.optimizer_value_head.state_dict())
    c.actor_optimizer.steps.fill_(9)
    for m in c.actor_optimizer.first:
        m.fill_(1)
    loss = c.adopt(1)
    assert torch.equal(c.vectors()[1:],before[1:])
    assert all(torch.equal(v,c.model.optimizer_value_head.state_dict()[k]) for k,v in value.items())
    if mode == "none":
        assert torch.equal(c.vectors(), before) and c.gate.count == 0
        assert c.actor_optimizer.steps.tolist() == [9]*4
    else:
        assert c.gate.count == 1 and c.gate.last_update == 1
        assert c.actor_optimizer.steps.tolist() == [0,9,9,9]
        assert all(m[0].count_nonzero() == 0 and torch.all(m[1:] == 1) for m in c.actor_optimizer.first)
        if mode == "copy":
            assert torch.equal(c.vectors()[0],before[1])
        else:
            assert np.isfinite(loss) and not torch.equal(c.vectors()[0],before[0])
        assert all(x.grad is None for x in c.teacher_conditions)


def test_diffusion_bc_reduces_noise_loss(tmp_path):
    torch.manual_seed(3)
    c = controller(tmp_path)
    policy = c.bank.export(c.vectors()[0])
    condition, action = torch.zeros(128,8), torch.zeros(128,2)
    steps, noise = torch.randint(2,(128,)), torch.randn(128,2)
    before = float(noise_prediction_loss(policy,condition,action,steps,noise))
    imitate(policy,condition,action,{"bc_steps":200,"bc_learning_rate":.01})
    after = float(noise_prediction_loss(policy,condition,action,steps,noise))
    assert after < before * .6


def test_no_c_and_zero_scale_still_updates_only_f(tmp_path):
    cfg = tiny(tmp_path, "no_c", **{"popexplore.actor_ppo_scale":0., "popexplore.update_rule":"none"})
    run_population(cfg, campaign_id="no-c", verbose=False)
    cp = torch.load(run_dir(tmp_path,"no-c") / "checkpoints/population.pt", weights_only=False)
    assert cp["population"]["actor_optimizer"]["steps"].tolist() == [2,0]
    vectors = torch.cat([p.reshape(2,-1) for k,p in cp["model"].items() if k.startswith("diffusion_policy.weights.")],1)
    assert not torch.equal(vectors[0], cp["population"]["original"])


def test_f_default_and_nested_validation(tmp_path):
    base = tiny(tmp_path,"no_f").to_dict()
    base["popexplore"].pop("inference_actor")
    base["popexplore"]["optimizer_agent"] = {"enabled":True}
    assert PopConfig(base).pop["inference_actor"] == "optimizer"
    for values in ({"window":1}, {"margin":-1}, {"confidence":1.}, {"adopt":"bad"}, {"bc_steps":False}, {"oops":1}):
        bad = copy.deepcopy(base)
        bad["popexplore"]["optimizer_agent"].update(values)
        with pytest.raises(ConfigError):
            PopConfig(bad)
    assert not tiny(tmp_path).pop["optimizer_agent"]["enabled"]


@pytest.mark.parametrize("mode", ["copy", "bc"])
def test_resume_through_actual_adoption(tmp_path, monkeypatch, mode):
    # Deterministic synthetic fitness makes the integration exercise real adoption,
    # without claiming a small environment must produce significant improvements.
    from sb_pomdp.popexplore import controller as module
    real_evaluate = module.evaluate_population
    def fitness(model, **kwargs):
        result = real_evaluate(model, **kwargs)
        if kwargs["seed"] >= 2_000_000:
            result["returns"][0] = 0
            result["returns"][1:] = 1
        return result
    monkeypatch.setattr(module, "evaluate_population", fitness)
    cfg = tiny(tmp_path,"full", **{"ppo.total_steps":48, "popexplore.selection_interval":2,
                   "popexplore.optimizer_agent.window":2, "popexplore.optimizer_agent.adopt":mode,
                   "ppo.bootstrap_on_truncation":True})
    run_population(cfg, campaign_id="whole", verbose=False)
    for _ in range(4):
        run_population(cfg, campaign_id="split", segment_updates=1, resume=True, verbose=False)
    a = torch.load(run_dir(tmp_path,"whole")/"checkpoints/population.pt",weights_only=False)
    b = torch.load(run_dir(tmp_path,"split")/"checkpoints/population.pt",weights_only=False)
    assert a["population"]["adoption_gate"]["count"] == b["population"]["adoption_gate"]["count"] == 2
    for name in a["model"]:
        assert torch.equal(a["model"][name],b["model"][name]), name
    assert torch.equal(a["torch_rng_state"],b["torch_rng_state"])
    for file in ("metrics.csv","population.csv","evaluation.csv"):
        ar,br = rows(run_dir(tmp_path,"whole")/file), rows(run_dir(tmp_path,"split")/file)
        for r in ar+br:
            r.pop("steps_per_second",None)
        assert ar == br


def test_analysis_partial_and_cross_task_pairs(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("analyzer", "scripts/popx_analyze.py")
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    cfg = tiny(tmp_path,"full", **{"experiment.eval_interval_updates":100})
    run_population(cfg,campaign_id="partial",segment_updates=1,verbose=False)
    out = tmp_path / "partial.md"
    result = analyzer.analyze([str(tmp_path/"campaigns")],out,plots=False)
    assert result["curve_rows"] == 0
    assert "評価未到達: 1" in out.read_text(encoding="utf-8")
    contrasts = [dict(group="scratch",mode="stochastic",arm="no_f",reference="full",task=t,
                     pairs=2, reference_mean=scale, paired_differences=json.dumps([[-1,10,scale],[-1,11,2*scale]]))
                 for t,scale in (("a",1),("b",10))]
    cross = analyzer.across_tasks(contrasts)[0]
    assert cross["task_count"] == 2 and cross["pairs"] == 2 and cross["difference"] == 1.5


def test_rollout_fitness_has_no_extra_environment_evaluation(tmp_path, monkeypatch):
    c = controller(tmp_path, **{"popexplore.fitness_from_rollout":True,
                               "popexplore.optimizer_agent.window":2, "popexplore.optimizer_agent.adopt":"copy"})
    def forbidden(*args, **kwargs):
        raise AssertionError("fast fitness must never launch additional episodes")
    monkeypatch.setattr("sb_pomdp.popexplore.controller.evaluate_population", forbidden)
    monkeypatch.setattr("sb_pomdp.popexplore.controller.evaluate", forbidden)
    rollout = SimpleNamespace(returns=torch.ones(2,4), old_values=torch.zeros(2,4))
    c.external = [torch.ones(4)]
    c.before_learning()
    result = c.finish_update(rollout, [])
    assert c.search.generation == 0 and result["selection_replaced"] == 0
    # Two completed rollout cohorts; no B replacement until the adoption window fills.
    c.cfg["selection_interval"] = 3
    for update in (2,3):
        c.begin_update(update,"ppo")
        c.external = [torch.ones(4)]
        c.finished = [[(float(i),np.array([i,0,0,0,1.]))] for i in range(4)]
        result = c.finish_update(rollout, [])
    assert c.gate.count == 1 and c.search.generation == 1
    assert np.isnan(result["optimizer_return_after"])
    assert all(r["fitness_source"] == "rollout_behavior" for r in rows(tmp_path / "population.csv"))


def test_pull_changes_only_search_mean(tmp_path, monkeypatch):
    c = controller(tmp_path, **{"popexplore.optimizer_agent.pull_to_optimizer":1.0})
    monkeypatch.setattr("sb_pomdp.popexplore.controller.evaluate_population", lambda *a, **k:dict(returns=np.ones((4,2)),descriptors=np.eye(4)))
    monkeypatch.setattr("sb_pomdp.popexplore.controller.evaluate", lambda *a, **k:({"mean_return":1.}, {}))
    original = c.vectors()[0].clone()
    c.external = [torch.ones(4)]
    c.finish_update(SimpleNamespace(returns=torch.ones(2,4),old_values=torch.zeros(2,4)), [])
    assert torch.equal(c.search.mean, original) and torch.equal(c.vectors()[0], original)
