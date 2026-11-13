"""Explicit population hooks for the otherwise unchanged sequence trainer."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..artifacts import CsvAppender
from ..initialization import load_bc_actor
from ..train import save_checkpoint, build_optimizer, source_sha256
from ..ppo import update_ppo as ordinary_ppo
from .actors import PopulationPolicy, OUBias, assignments
# Single-actor diagnostics use the batched evaluator (one horizon, not one per episode).
from .evaluation import evaluate_population, evaluate_batched as evaluate
from .rnd import RND
from .selection import PopulationSearch, ActorAdam, rank_correlation
from .optimizer_agent import AdoptionGate, imitate


DIAGNOSTIC_FIELDS = (
    "fitness_update", "fitness_return_std", "fitness_noise_se", "rank_stability",
    "selection_replaced", "selection_before_return", "selection_after_return",
    "weight_spread", "descriptor_spread", "inference_distance", "bc_projection", "bc_cosine",
    "ou_rho", "ou_effective_tau", "rnd_reward_mean", "rnd_external_ratio", "rnd_loss",
    "value_explained_variance", "actor_ppo_fraction", "population_generation",
    "optimizer_adoptions", "optimizer_adopted", "optimizer_adoption_update", "optimizer_candidate",
    "optimizer_pair_count", "optimizer_difference", "optimizer_lower_bound", "optimizer_bc_loss",
    "optimizer_return_before", "optimizer_return_after", "optimizer_return_pre_ppo",
    "optimizer_explorer_best", "optimizer_fitness_return", "optimizer_ppo_return_change",
)
POPULATION_FIELDS = ("update", "ppo_update", "generation", "individual", "return_mean", "return_std",
                     "episodes", "descriptor", "fitness_source", "selected", "parent", "replaced",
                     "weight_distance", "actor_ppo_enabled", "optimizer_adoptions", "optimizer_adopted",
                     "optimizer_adoption_update", "optimizer_candidate", "optimizer_difference", "optimizer_lower_bound")
MEMBER_FIELDS = ("update", "global_step", "mode", "actor", "evaluation_seed", "mean_return",
                 "std_return", "mean_length", "action_saturation_fraction")


def source_hash():
    h = hashlib.sha256()
    h.update(source_sha256().encode())
    seedbatch = Path(__file__).parent.parent / "seedbatch"
    for name in ("backend.py", "attention.py"):
        h.update((seedbatch / name).read_bytes())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


class Controller:
    def __init__(self, model, resolved, task, seed, artifacts, environment, device, resume=False):
        self.model, self.resolved, self.cfg = model, resolved, resolved["popexplore"]
        self.task, self.seed, self.artifacts, self.device = task, seed, artifacts, device
        self.n = self.cfg["population_size"]
        self.fcfg = self.cfg["optimizer_agent"]
        self.f_enabled = self.fcfg["enabled"]
        self.first_explorer = int(self.f_enabled)
        self.gate = AdoptionGate(self.fcfg, self.n) if self.f_enabled else None
        if self.f_enabled:
            # Separate external-only critic; no intrinsic targets ever reach it.
            model.optimizer_value_head = copy.deepcopy(model.value_head)
        self.ids = assignments(resolved["ppo"]["num_envs"], self.n, device)
        self.original = torch.cat([p.detach().flatten() for p in model.diffusion_policy.parameters()]).clone()
        self.bank = None
        if self.n > 1:
            self.bank = PopulationPolicy(model.diffusion_policy, self.n, len(self.ids), self.cfg["init_std"])
            model.diffusion_policy = self.bank
            if self.f_enabled:
                vectors = self.bank.vectors().detach().clone()
                vectors[0] = self.original
                self.bank.set_vectors(vectors)
                self.bank.start_episodes(torch.ones_like(self.ids, dtype=torch.bool))
        elif self.cfg["init_std"]:
            with torch.no_grad():
                for p in model.diffusion_policy.parameters():
                    p.add_(torch.randn_like(p) * self.cfg["init_std"])
        self.search = PopulationSearch(self.vectors()[self.first_explorer:], self.cfg)
        self.actor_optimizer = ActorAdam(self.bank.weights, resolved["ppo"]) if self.bank is not None else None
        self.best_vector = self.vectors()[0].detach().clone()
        self.bc_direction = None
        path = self.cfg["bc_actor_path"]
        if not path and task == "disk_tag" and self.cfg["checkpoint_seed"] >= 0:
            candidates = sorted(Path("assets/bc_actors").glob(f"*_{self.cfg['checkpoint_seed']}_*.pt"))
            if len(candidates) == 1:
                path = str(candidates[0])
        if path and not resume:
            proto = copy.deepcopy(self.bank.prototype if self.bank else model.diffusion_policy)
            load_bc_actor(proto, path)
            self.bc_direction = torch.cat([p.detach().flatten() for p in proto.parameters()]) - self.original
        self.ou = OUBias(len(self.ids), model.diffusion_policy.action_dim,
                         self.cfg["ou_std"], self.cfg["ou_tau"], device) if self.cfg["ou_std"] else None
        self.rnd = None
        if self.cfg["rnd_beta"]:
            input_dim = environment.observation_dim if self.cfg["rnd_input"] == "observation" else resolved["model"]["d_model"]
            self.rnd = RND(input_dim, self.cfg, device)
        self.ppo_updates, self.update, self.fitness_update = 0, 0, 0
        self.actor_enabled = torch.ones(self.n, dtype=torch.bool, device=device)
        self.active_actors = torch.zeros_like(self.actor_enabled)
        self.pending_returns = np.zeros(len(self.ids))
        self.pending_lengths = np.zeros(len(self.ids), dtype=int)
        self.pending_sums = np.zeros((len(self.ids), model.diffusion_policy.action_dim))
        self.pending_squares = np.zeros_like(self.pending_sums)

    def vectors(self):
        if self.bank is not None:
            return self.bank.vectors().detach()
        return torch.cat([p.detach().flatten() for p in self.model.diffusion_policy.parameters()])[None]

    def exclude_actor_parameters(self, optimizer):
        if self.bank is not None:
            actor = set(self.bank.parameters())
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if p not in actor]

    def zero_actor_grad(self):
        self.active_actors.zero_()
        if self.actor_optimizer is not None:
            self.actor_optimizer.zero_grad()

    def step_actors(self):
        if self.actor_optimizer is not None:
            self.actor_optimizer.step(self.active_actors)

    def suppress_disabled_singleton(self):
        if self.bank is None and not bool(self.actor_enabled[0]):
            for p in self.model.diffusion_policy.parameters():
                p.grad = None

    def begin_update(self, update, phase):
        self.update, self.phase = update, phase
        if phase == "ppo":
            self.ppo_updates += 1
        size = self.n - self.first_explorer
        count = math.ceil(size * self.cfg["actor_ppo_fraction"])
        self.actor_enabled = (torch.arange(self.n, device=self.device) - self.first_explorer - self.ppo_updates) % size < count
        if not self.cfg["actor_ppo_scale"]:
            self.actor_enabled.zero_()
        if self.f_enabled:
            self.actor_enabled[0] = True
        self.teacher_conditions, self.teacher_actions = [], []
        self.pre_ppo_return = float("nan")
        self.rnd_inputs, self.intrinsic, self.external = [], [], []
        self.finished = [[] for _ in range(self.n)]
        if self.ou is not None:
            self.ou.xy = self.ou.xx = self.ou.yy = 0.0
            self.ou.pairs = 0

    def start_step(self, initial):
        if self.bank is not None:
            self.bank.start_episodes(initial)

    def perturb_action(self, action, chain, initial, phase):
        if self.ou is None or phase == "pretrain":
            return action
        bias = self.ou.step(initial, self.ids != 0 if self.f_enabled else None)
        p = self.model.diffusion_policy
        # The stored diffusion chain is unchanged. Given exogenous bias b_t,
        # a_t = scale*tanh(z_t+b_t)+offset has a theta-independent Jacobian.
        return torch.tanh(chain[:, -1] + bias) * p.action_scale + p.action_bias

    def observe(self, action, reward, done, phase):
        p = self.model.diffusion_policy
        normalized = ((action - p.action_bias) / p.action_scale).detach().cpu().numpy()
        self.pending_returns += reward
        self.pending_lengths += 1
        self.pending_sums += normalized
        self.pending_squares += normalized**2
        env_per_actor = len(self.ids) // self.n
        horizon = self.resolved["environment"]["tasks"][self.task]["horizon"]
        for e in np.flatnonzero(done):
            length = self.pending_lengths[e]
            mean = self.pending_sums[e] / length
            var = np.maximum(0, self.pending_squares[e] / length - mean**2)
            if phase != "pretrain":
                self.finished[e // env_per_actor].append((self.pending_returns[e], np.r_[mean, var, length / horizon]))
            self.pending_returns[e] = self.pending_lengths[e] = 0
            self.pending_sums[e] = self.pending_squares[e] = 0

    def reward(self, external, observation, sink, phase):
        self.external.append(external.detach())
        if self.rnd is None or phase == "pretrain":
            return external
        inputs = observation if self.cfg["rnd_input"] == "observation" else sink[0][0]
        self.rnd_inputs.append(inputs.detach())
        intrinsic = self.cfg["rnd_beta"] * self.rnd.reward(inputs)
        if self.f_enabled:
            intrinsic = intrinsic.masked_fill(self.ids == 0, 0)
        self.intrinsic.append(intrinsic)
        return external + intrinsic

    def values(self, condition, ids, ordinary=None):
        if ordinary is None:
            ordinary = self.model.value(condition)
        if not self.f_enabled:
            return ordinary
        external = self.model.optimizer_value_head(condition).squeeze(-1)
        return torch.where(ids == 0, external, ordinary)

    def save_teacher(self, sink, action):
        if self.f_enabled and self.fcfg["adopt"] == "bc":
            self.teacher_conditions.append(sink[0][0].detach().clone())
            self.teacher_actions.append(action.detach().clone())

    def before_learning(self):
        self.pre_ppo_return = float("nan")
        if (self.f_enabled and self.phase == "ppo" and not self.cfg["fitness_from_rollout"]
                and self.gate.followup):
            # Same current-update seeds before/after PPO isolate immediate undoing.
            # Only the PPO update right after an adoption needs this diagnostic.
            args = self.eval_args(self.cfg["fitness_episodes"], self.seed + 2_000_000 + self.ppo_updates * 1000, False)
            summary, _ = evaluate(self.inference_model("optimizer"), **args)
            self.pre_ppo_return = summary["mean_return"]

    def truncation_values(self, model, model_config, *, infos, particles, action_features, device, value_transform):
        from ..train import (truncated_step_indices, final_step_features, BeliefContext,
                             _sample_belief, encode_conditions, control_conditions)
        values = torch.zeros(particles.shape[0], dtype=torch.float32, device=device)
        indices = truncated_step_indices(infos)
        if not indices:
            return values
        selector = torch.as_tensor(indices, dtype=torch.long, device=device)
        context = BeliefContext(
            observation=final_step_features(infos, indices, name="final_observation", device=device),
            previous_particles=particles.detach().index_select(0, selector).clone(),
            previous_action=action_features.detach().index_select(0, selector).clone(),
            initial=torch.zeros(len(indices), dtype=torch.bool, device=device))
        with torch.no_grad():
            final_particles, scores = _sample_belief(model, context, model_config)
            policy, critic = encode_conditions(model, final_particles, scores)
            _, critic = control_conditions(model, policy, critic, context.observation)
            bootstrap = value_transform.to_raw(self.values(critic, self.ids[selector]))
        values.index_copy_(0, selector, bootstrap.detach().to(values))
        return values

    def update_ppo(self, *args, **kwargs):
        if self.bank is None and self.cfg["actor_ppo_fraction"] == 1 and self.cfg["actor_ppo_scale"] == 1:
            return ordinary_ppo(*args, **kwargs)
        from .learning import update_ppo
        return update_ppo(*args, **kwargs)

    def eval_args(self, episodes, seed, deterministic):
        return dict(task=self.task, environment_config=self.resolved["environment"]["tasks"][self.task],
                    model_config=self.resolved["model"], episodes=episodes, seed=seed,
                    device=self.device, deterministic=deterministic)

    def mean_vector(self):
        if self.cfg["update_rule"] in {"cem", "qd", "es"}:
            return self.search.mean
        return self.vectors()[self.first_explorer:].mean(0)

    def refresh_archive(self, args):
        """Re-score historical actors under CURRENT belief and fitness seeds."""
        if not self.search.archive:
            return
        current = self.vectors().clone()
        entries = list(self.search.archive.values())
        refreshed = {}
        try:
            for start in range(0, len(entries), self.n):
                group = entries[start:start+self.n]
                vectors = torch.stack([entry["vector"].to(current) for entry in group])
                if len(group) < self.n:
                    vectors = torch.cat((vectors, current[len(group):]))
                self.bank.set_vectors(vectors)
                result = evaluate_population(self.model, **args)
                for i, entry in enumerate(group):
                    entry = {**entry, "fitness": float(result["returns"][i].mean()),
                             "descriptor": result["descriptors"][i]}
                    key = tuple(np.floor(entry["descriptor"] / self.cfg["qd_cell_size"]).astype(int))
                    if key not in refreshed or entry["fitness"] > refreshed[key]["fitness"]:
                        refreshed[key] = entry
            self.search.archive = refreshed
        finally:
            self.bank.set_vectors(current)

    def inference_model(self, kind=None):
        if self.bank is None:
            return self.model
        # Share the trained belief/value objects; only the inference actor is copied.
        result = copy.copy(self.model)
        result._modules = self.model._modules.copy()
        kind = kind or self.cfg["inference_actor"]
        vector = self.vectors()[0] if kind == "optimizer" else self.best_vector if kind == "best" else self.mean_vector()
        result.diffusion_policy = self.bank.export(vector)
        if self.f_enabled:
            result._modules.pop("optimizer_value_head")
            if kind == "optimizer":
                result.value_head = self.model.optimizer_value_head
        return result

    def adopt(self, candidate):
        vectors = self.vectors().clone()
        loss = float("nan")
        if self.fcfg["adopt"] == "copy":
            vectors[0] = vectors[candidate]
        elif self.fcfg["adopt"] == "bc":
            teacher_mask = self.ids == candidate
            condition = torch.stack(self.teacher_conditions)[:, teacher_mask].flatten(0, 1)
            actions = torch.stack(self.teacher_actions)[:, teacher_mask].flatten(0, 1)
            policy = self.bank.export(vectors[0])
            loss = imitate(policy, condition, actions, self.fcfg)
            vectors[0] = torch.cat([p.detach().flatten() for p in policy.parameters()])
        else:
            return loss
        self.bank.set_vectors(vectors)
        self.actor_optimizer.reset(torch.arange(self.n, device=self.device) == 0)
        self.gate.count += 1
        self.gate.last_update = self.update
        self.gate.reset()
        return loss

    def finish_update(self, rollout, update_returns):
        d = {key: float("nan") for key in DIAGNOSTIC_FIELDS}
        n = self.n
        selected, parents, changed = np.zeros(n, bool), np.arange(n), np.zeros(n, bool)
        fitness = None
        before_vectors = self.vectors().clone()
        selection_due = self.ppo_updates % self.cfg["selection_interval"] == 0
        if self.phase == "ppo" and n > 1 and (selection_due or self.f_enabled):
            seed = self.seed + 2_000_000 + self.ppo_updates * 1000
            args = self.eval_args(self.cfg["fitness_episodes"], seed, False)
            rollout_fitness = self.cfg["fitness_from_rollout"]
            if rollout_fitness:
                # Deliberate fast/approximate option: these are executed episode
                # returns, possibly old snapshots + OU, NOT fixed-current-policy tests.
                desc_dim = 2 * self.bank.action_dim + 1
                fitness = dict(returns=[np.asarray([r[0] for r in items]) for items in self.finished],
                               descriptors=np.asarray([np.mean([r[1] for r in items], axis=0) if items
                                                       else np.full(desc_dim, np.nan) for items in self.finished]))
                scores = np.asarray([r.mean() if len(r) else np.nan for r in fitness["returns"]])
                repeat = None
            else:
                fitness = evaluate_population(self.model, **args)
                repeat = evaluate_population(self.model, **{**args, "seed": seed + 500}) if selection_due else None
                scores = fitness["returns"].mean(1)
            if np.isfinite(scores).any():
                self.best_vector = before_vectors[int(np.nanargmax(scores))].clone()
            if repeat is not None:
                d["rank_stability"] = rank_correlation(scores, repeat["returns"].mean(1))
                d["fitness_noise_se"] = float(np.sqrt(np.mean((scores-repeat["returns"].mean(1))**2) / 2))
            finite_scores = scores[np.isfinite(scores)]
            if len(finite_scores):
                d["fitness_return_std"] = float(finite_scores.std())
                exploration_scores = scores[self.first_explorer:]
                if np.isfinite(exploration_scores).any():
                    d["selection_before_return"] = float(exploration_scores[np.isfinite(exploration_scores)].mean())
            if self.f_enabled:
                decision = self.gate.observe(self.ppo_updates, scores)
                summary = {"mean_return":float("nan")}
                adopting = decision["eligible"] and self.fcfg["adopt"] != "none"
                # F's own before/after evaluation is a diagnostic of adoption events:
                # run it when F is about to adopt, and in the PPO update after one.
                if not rollout_fitness and (adopting or self.gate.followup):
                    summary, _ = evaluate(self.inference_model("optimizer"), **args)
                self.gate.followup = False
                before = summary["mean_return"]
                d.update(optimizer_candidate=decision["candidate"], optimizer_pair_count=decision["pairs"],
                         optimizer_difference=decision["difference"], optimizer_lower_bound=decision["lower"],
                         optimizer_return_before=before, optimizer_return_pre_ppo=self.pre_ppo_return,
                         optimizer_ppo_return_change=before-self.pre_ppo_return,
                         optimizer_explorer_best=float(np.nanmax(scores[1:])) if np.isfinite(scores[1:]).any() else float("nan"),
                         optimizer_fitness_return=float(scores[0]),
                         optimizer_adopted=0)
                if adopting:
                    d["optimizer_bc_loss"] = self.adopt(decision["candidate"])
                    d["optimizer_adopted"] = 1
                    self.gate.followup = True
                    if not rollout_fitness:
                        summary, _ = evaluate(self.inference_model("optimizer"), **args)
                d["optimizer_return_after"] = summary["mean_return"]
            # No completed episodes => skip, never impute zero or secretly evaluate.
            if selection_due and np.isfinite(scores[self.first_explorer:]).all():
                if self.cfg["update_rule"] == "qd" and not rollout_fitness:
                    self.refresh_archive(args)
                offset = self.first_explorer
                vectors, sel, par, chg = self.search.update(
                    before_vectors[offset:], scores[offset:], fitness["descriptors"][offset:])
                selected[offset:], changed[offset:] = sel, chg
                parents[offset:] = np.where(par >= 0, par + offset, par)
                combined = self.vectors().clone()
                combined[offset:] = vectors
                self.bank.set_vectors(combined)
                self.actor_optimizer.reset(torch.as_tensor(changed, device=self.device))
                if self.f_enabled:
                    self.gate.reset(changed)
                    if self.cfg["update_rule"] in {"cem", "qd", "es"}:
                        self.search.mean.lerp_(combined[0], self.fcfg["pull_to_optimizer"])
                if not rollout_fitness:
                    after = evaluate_population(self.model, **args) if changed.any() else fitness
                    d["selection_after_return"] = float(after["returns"][self.first_explorer:].mean())
                    after_scores = after["returns"].mean(1)
                    self.best_vector = self.vectors()[int(after_scores.argmax())].clone()
            if len(finite_scores):
                self.fitness_update = self.update
        if self.f_enabled:
            d.update(optimizer_adoptions=self.gate.count, optimizer_adoption_update=self.gate.last_update)
        vectors = self.vectors()
        center = vectors.mean(0)
        distance = (vectors - center).norm(dim=1)
        d.update(fitness_update=self.fitness_update, selection_replaced=int(changed.sum()),
                 weight_spread=float(distance.mean()), actor_ppo_fraction=float(self.actor_enabled.float().mean()),
                 population_generation=self.search.generation)
        inferred = (self.vectors()[0] if self.cfg["inference_actor"] == "optimizer" else
                    self.best_vector if self.cfg["inference_actor"] == "best" and self.n > 1 else self.mean_vector())
        delta = inferred - self.original
        d["inference_distance"] = float(delta.norm())
        if self.bc_direction is not None:
            denom = self.bc_direction.square().sum()
            if float(denom) > 0:
                d["bc_projection"] = float(delta.dot(self.bc_direction) / denom)
                if float(delta.norm()) > 0:
                    d["bc_cosine"] = float(delta.dot(self.bc_direction) / (delta.norm() * denom.sqrt()))
        if self.ou is not None:
            d["ou_rho"], d["ou_effective_tau"] = self.ou.correlation()
        d["rnd_reward_mean"] = float(torch.cat(self.intrinsic).mean()) if self.intrinsic else 0.0
        external = float(torch.cat(self.external).abs().mean())
        d["rnd_external_ratio"] = d["rnd_reward_mean"] / max(external, 1e-8)
        d["rnd_loss"] = self.rnd.update_predictor(torch.cat(self.rnd_inputs)) if self.rnd_inputs else 0.0
        target, old = rollout.returns.detach(), rollout.old_values.detach()
        variance = target.var(unbiased=False)
        d["value_explained_variance"] = float(1 - (target-old).var(unbiased=False)/variance) if float(variance) > 1e-12 else float("nan")
        rows, descriptors = [], []
        for i in range(n):
            results = fitness["returns"][i] if fitness is not None else np.asarray([r[0] for r in self.finished[i]])
            desc = fitness["descriptors"][i] if fitness is not None else (np.mean([r[1] for r in self.finished[i]], axis=0) if self.finished[i] else None)
            if desc is not None:
                descriptors.append(desc)
            parent = str(int(parents[i]))
            if parents[i] <= -2:
                cell = list(self.search.archive)[-2-int(parents[i])]
                parent = "archive:" + json.dumps([int(x) for x in cell])
            rows.append(dict(update=self.update, ppo_update=self.ppo_updates, generation=self.search.generation-int(changed.any()),
                             individual=i, return_mean=float(results.mean()) if len(results) else float("nan"),
                             return_std=float(results.std()) if len(results) else float("nan"), episodes=len(results),
                             descriptor=json.dumps(desc.tolist()) if desc is not None else "",
                             fitness_source=("rollout_behavior" if self.cfg["fitness_from_rollout"] else "fixed_policy_test") if fitness is not None else "training_episodes",
                             selected=int(selected[i]), parent=parent, replaced=int(changed[i]),
                             weight_distance=float((before_vectors[i]-before_vectors.mean(0)).norm()),
                             actor_ppo_enabled=int(self.actor_enabled[i]),
                             **{k: d[k] for k in POPULATION_FIELDS if k.startswith("optimizer_")}))
        if len(descriptors) > 1:
            d["descriptor_spread"] = float(np.mean(np.linalg.norm(np.asarray(descriptors)-np.mean(descriptors, axis=0), axis=1)))
        CsvAppender(self.artifacts.path / "population.csv", POPULATION_FIELDS).append_many(rows)
        return d

    def evaluate_members(self, update, global_step):
        rows = []
        episodes = self.resolved["experiment"]["eval_episodes"]
        seed = self.seed + 1_000_000
        for mode, deterministic in (("mean_chain", True), ("stochastic", False)):
            args = self.eval_args(episodes, seed, deterministic)
            result = evaluate_population(self.model, **args)
            common = dict(update=update, global_step=global_step, mode=mode, evaluation_seed=seed)
            for i in range(self.n):
                rows.append({**common, "actor": str(i), "mean_return": float(result["returns"][i].mean()),
                             "std_return": float(result["returns"][i].std()), "mean_length": float(result["lengths"][i].mean()),
                             "action_saturation_fraction": float(result["saturation"][i])})
            for kind in (("mean", "best", "optimizer") if self.f_enabled else ("mean", "best")):
                summary, _ = evaluate(self.inference_model(kind), **args)
                rows.append({**common, "actor": kind, **summary})
        CsvAppender(self.artifacts.path / "population_evaluation.csv", MEMBER_FIELDS).append_many(rows)

    def state_dict(self):
        return dict(source_hash=source_hash(), original=self.original, best_vector=self.best_vector,
                    bc_direction=self.bc_direction, search=vars(self.search), ppo_updates=self.ppo_updates,
                    fitness_update=self.fitness_update, episode_weights=None if self.bank is None else self.bank.episode_weights,
                    actor_optimizer=None if self.actor_optimizer is None else self.actor_optimizer.state_dict(),
                    ou=None if self.ou is None else vars(self.ou), rnd=None if self.rnd is None else self.rnd.training_state(),
                    pending_returns=self.pending_returns, pending_lengths=self.pending_lengths,
                    pending_sums=self.pending_sums, pending_squares=self.pending_squares,
                    adoption_gate=None if self.gate is None else vars(self.gate))

    def restore(self, state):
        if state["source_hash"] != source_hash():
            raise ValueError("popexplore source changed since checkpoint; use the original source to resume")
        for name in ("original", "best_vector", "bc_direction"):
            value = state[name]
            setattr(self, name, value.to(self.device) if value is not None else None)
        self.search.__dict__.update(state["search"])
        self.search.mean, self.search.std = self.search.mean.to(self.device), self.search.std.to(self.device)
        self.ppo_updates, self.fitness_update = state["ppo_updates"], state["fitness_update"]
        if self.gate is not None:
            self.gate.__dict__.update(state["adoption_gate"])
        if self.bank is not None:
            self.bank.episode_weights = [p.to(self.device) for p in state["episode_weights"]]
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if self.ou is not None:
            self.ou.__dict__.update(state["ou"])
            self.ou.value = self.ou.value.to(self.device)
        if self.rnd is not None:
            self.rnd.restore(state["rnd"])
        for name in ("pending_returns", "pending_lengths", "pending_sums", "pending_squares"):
            setattr(self, name, state[name].copy())

    def truncate(self, update):
        for name in ("population", "population_evaluation", "timing"):
            self.artifacts.truncate_csv(lambda row: int(row["update"]) <= update, name=name)

    def save_checkpoint(self, payload, path):
        # Authoritative resume state is separate from the ordinary inference model.
        if path.stem == "latest":
            save_checkpoint(payload, path.with_name("population.pt"))
        inference = self.inference_model()
        exported = dict(payload)
        exported.pop("population", None)
        exported.pop("trainer_state", None)  # fail clearly if passed to train --resume
        exported["config"] = {k: v for k, v in payload["config"].items() if k != "popexplore"}
        exported["model"] = inference.state_dict()
        exported["optimizer"] = build_optimizer(inference, self.resolved["ppo"]).state_dict()
        exported["popexplore_inference_only"] = True
        save_checkpoint(exported, path)
