"""CPU parameter and matmul-FLOP accounting for the production GRU/light arms.

FLOPs count multiply and add as two, excluding elementwise operations. Attention
is explicitly expanded on a private model to avoid fused inference undercount.
This is rollout acting cost, not backward/replay/fitness or campaign runtime.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from sb_pomdp.envs import make_env
from sb_pomdp.popexplore.config import load_config
from sb_pomdp.popexplore.representation import make_model, heads, actor, is_gru
from sb_pomdp.seedbatch.attention import use_batched_attention
from sb_pomdp.seedbatch.backend import functional_kernels
from sb_pomdp.train import BeliefContext, _sample_belief, belief_latent_dim
from sb_pomdp.ppo import encode_conditions


def count(function):
    with torch.no_grad(), functional_kernels(), FlopCounterMode(display=False) as counter:
        output = function()
    operations = {str(k):v for k,v in counter.get_flop_counts().get("Global", {}).items()}
    return output, counter.get_total_flops(), operations


def measure(task, arm):
    cfg = load_config(f"config/popx_scratch_{task}_{arm}.json")
    data = cfg.base.to_dict()
    env_task = data["experiment"]["tasks"][0]
    env = make_env(env_task, cfg.environment_for_task(env_task), seed=10)
    torch.manual_seed(10)
    model = make_model(env, data, torch.device("cpu")).eval()
    use_batched_attention(model)
    policy, head = actor(model), heads(model)
    actor_count = sum(p.numel() for p in policy.parameters())
    value_count = sum(p.numel() for p in head.value_head.parameters())
    prediction_count = sum(p.numel() for p in model.prediction_predictor.parameters())
    ordinary_count = sum(p.numel() for p in model.parameters())
    f = cfg.pop["optimizer_agent"]["enabled"]
    n = cfg.pop["population_size"]
    observation, _ = env.reset(seed=10)
    observation = torch.as_tensor(observation)[None]
    previous_action = torch.zeros(1, env.action_spec.feature_dim)
    inputs = torch.cat((observation, previous_action), -1)
    flops, ops = {}, {}
    for initial in (True, False):
        if is_gru(model):
            recurrent = model.initial_hidden(1)
            def representation():
                return model.step(inputs, recurrent, torch.tensor([initial]))[0]
        else:
            context = BeliefContext(observation, torch.zeros(1, data["model"]["num_particles"],
                belief_latent_dim(env, data["model"])), previous_action, torch.tensor([initial]))
            def representation():
                particles, scores = _sample_belief(model, context, data["model"])
                return encode_conditions(model, particles, scores)[0]
        condition, amount, operators = count(representation)
        label = "initial" if initial else "recursive"
        flops[f"representation_{label}"] = amount
        ops[f"representation_{label}"] = operators
    # Native baseline/score rollout both compute chain entropy as well as sample.
    _, flops["actor_sample"], ops["actor_sample"] = count(lambda: policy.sample(condition, deterministic=False))
    _, flops["actor_entropy"], ops["actor_entropy"] = count(lambda: policy.chain_entropy(condition))
    _, flops["value"], ops["value"] = count(lambda: head.value(condition))
    flops["external_F_value"] = flops["value"] if f else 0
    for label in ("initial", "recursive"):
        flops[f"rollout_{label}"] = sum(flops[k] for k in
            (f"representation_{label}", "actor_sample", "actor_entropy", "value", "external_F_value"))
    shared = ordinary_count - actor_count + (value_count if f else 0)
    # RND parameters are checkpointed separately from the actor-critic model.
    from sb_pomdp.popexplore.rnd import RND
    rnd = RND(data["model"]["d_model"], cfg.pop, "cpu") if cfg.pop["rnd_beta"] else None
    rnd_parameters = sum(p.numel() for p in rnd.parameters()) if rnd is not None else 0
    return dict(task=task, arm=arm, population=n, actor_parameters_each=actor_count,
        actor_parameters_population=n * actor_count, representation_parameters=ordinary_count-actor_count-value_count-prediction_count,
        ordinary_value_parameters=value_count, prediction_parameters=prediction_count,
        external_F_value_parameters=value_count if f else 0, shared_parameters=shared,
        training_model_parameters=shared+n*actor_count, inference_parameters=ordinary_count,
        rnd_parameters=rnd_parameters, flop_per_environment_step=flops, counted_operators=ops,
        comparison=data["comparison"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    results = [measure(task, arm) for task in ("carflag", "mirror2", "tagr")
               for arm in ("ppo", "full", "gru_ppo", "gru_full")]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(torch=torch.__version__, device="cpu", batch=1,
        convention="matmul FLOPs: multiply+add=2; expanded attention; excludes elementwise, RND, prediction forward, PPO backward/replay, fitness",
        results=results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
