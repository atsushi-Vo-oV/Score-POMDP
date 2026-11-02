"""Small option-C campaign with separate warm-up and measured PPO updates.

Run with the worktree Python, or on the server with --device cuda --cuda-graphs.
Every setting runs in a fresh process, with the same config, seed and overrides.
The default is one prediction pretrain, one PPO warm-up, then three measured PPO
updates. Later signatures can still compile; measured does not promise steady state.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def long_path(path):
    value = str(path.resolve())
    return "\\\\?\\" + value if sys.platform == "win32" and not value.startswith("\\\\?\\") else value


def summarize_phases(log, warmup_updates, measured_updates):
    phases = {}
    for phase, update, seconds in re.findall(
        r"phase=(\w+) status=end update=(\d+)/\d+ seconds=([\d.]+)", log
    ):
        phases.setdefault(phase, []).append({"update": int(update), "seconds": float(seconds)})
    updates = phases.get("ppo_update", [])
    expected = 1 + warmup_updates + measured_updates
    # The trainer uses the ppo_update timer for prediction pretraining too.
    # Require the complete schedule so a zero-exit early stop is not a timing.
    if [item["update"] for item in updates] != list(range(1, expected + 1)):
        raise RuntimeError(
            f"incomplete benchmark: expected {expected} completed update timers "
            f"(1 pretrain + {warmup_updates} PPO warm-up + {measured_updates} measured PPO), "
            f"got {[item['update'] for item in updates]}"
        )
    predictions = re.findall(r"prediction phase=(\w+) update=(\d+)", log)
    if predictions != [("pretrain", "1")]:
        raise RuntimeError(f"unexpected prediction schedule: {predictions}")
    ppo = updates[1:]
    measured = ppo[warmup_updates:]
    return phases, {
        "pretrain": updates[:1], "warmup": ppo[:warmup_updates], "measured": measured,
        "measured_mean_seconds": sum(item["seconds"] for item in measured) / len(measured),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-graphs", action="store_true", help="include analytic+CUDA Graphs (compile off)")
    parser.add_argument("--setting", choices=("autograd", "analytic", "inductor", "graphs"))
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--measured-updates", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "agent" / "option-c-benchmark")
    args = parser.parse_args()
    if args.warmup_updates < 1 or args.measured_updates < 3:
        parser.error("require >= 1 PPO warm-up update and >= 3 measured PPO updates")
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires a CUDA-enabled PyTorch installation and GPU")
    if args.setting:
        try:
            returncode = run_setting(args)
            if returncode not in (None, 0, 77):
                print(f"ERROR: setting={args.setting}: training returned code {returncode}",
                      file=sys.stderr, flush=True)
            return returncode
        except Exception as error:
            print(f"ERROR: setting={args.setting}: {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
            raise
    settings = ["autograd", "analytic", "inductor"]
    if args.cuda_graphs:
        settings.append("graphs")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"platform": platform.platform(), "torch": torch.__version__, "device": args.device,
               "threads": 1, "timing": "end-to-end plus separate PPO warm-up/measured timings; "
               "later signatures may still compile/capture",
               "warmup_updates": args.warmup_updates, "measured_updates": args.measured_updates,
               "settings": {}}
    checkpoints = {}
    failed = False
    for setting in settings:
        log_path = args.output / f"{setting}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--device", args.device,
                   "--setting", setting, "--output", str(args.output),
                   "--warmup-updates", str(args.warmup_updates),
                   "--measured-updates", str(args.measured_updates)]
        started = time.perf_counter()
        print(f"START: setting={setting}; log={log_path}", flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as stream, subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            ) as process:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(f"[{setting}] {line}", end="", flush=True)
                returncode = process.wait()
            log = log_path.read_text(encoding="utf-8")
            if returncode == 77:
                reason = next(line for line in log.splitlines() if line.startswith("SKIP:"))
                summary["settings"][setting] = {"skipped": reason}
                continue
            if returncode:
                raise RuntimeError(f"child exited with code {returncode}; see {log_path}")
            phases, ppo = summarize_phases(log, args.warmup_updates, args.measured_updates)
            artifact = Path(long_path(args.output / setting))
            checkpoint_path = max(artifact.rglob("latest.pt"), key=lambda path: path.stat().st_mtime)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if checkpoint["update"] != 1 + args.warmup_updates + args.measured_updates:
                raise RuntimeError(f"incomplete checkpoint at update {checkpoint['update']}")
            checkpoints[setting] = checkpoint
            summary["settings"][setting] = {
                "wall_seconds": time.perf_counter() - started, "phases": phases, "ppo": ppo,
            }
            print(f"DONE: {setting}: PPO {json.dumps(ppo)}", flush=True)
        except Exception as error:  # noqa: BLE001 - record each isolated setting's failure
            failed = True
            summary["settings"][setting] = {"error": str(error), "log": str(log_path)}
            print(f"ERROR: setting={setting}: {error}; log={log_path}", file=sys.stderr, flush=True)
        finally:
            (args.output / "timings.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    reference = checkpoints.get("autograd")
    if reference is not None:
        for setting, checkpoint in checkpoints.items():
            differences = {name: (value - checkpoint["model"][name]).abs().max().item()
                           for name, value in reference["model"].items() if value.numel()}
            summary["settings"][setting]["checkpoint_comparison"] = {
                "model_max_abs": max(differences.values()),
                "largest_parameter": max(differences, key=differences.get),
                "torch_rng_equal": torch.equal(reference["torch_rng_state"], checkpoint["torch_rng_state"]),
                "python_rng_equal": reference["python_rng_state"] == checkpoint["python_rng_state"],
            }
    (args.output / "timings.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


def run_setting(args):
    if args.setting == "graphs" and args.device != "cuda":
        print("SKIP: CUDA Graphs requires --device cuda; CPU flag is a warned no-op", flush=True)
        return 77
    if args.setting == "inductor":
        # Probe actual forward + backward compilation. Missing CPU compilers are
        # expected locally; all other failures are actionable, never called timings.
        try:
            if args.device == "cpu":
                from torch._inductor.codecache import cpp_compiler
                cpp_compiler()
            value = torch.ones(8, device=args.device, requires_grad=True)
            compiled = torch.compile(lambda x: x.sin().square().sum(), fullgraph=True,
                                     dynamic=False, options={"triton.cudagraphs": False})
            compiled(value).backward()
        except Exception as error:
            if args.device == "cpu" and "InvalidCxxCompiler" in str(type(error)) + str(error):
                print("SKIP: Inductor CPU unavailable: InvalidCxxCompiler (no C++ compiler)")
                return 77
            raise
    from sb_pomdp.compare import main as compare

    command = ["--config", "config/trysvgd_tagr_tf-ula.json", "--methods", "score_transformer",
               "--seeds", "10", "--label", f"option-c-{args.setting}"]
    for override in (
        f"experiment.device={args.device}", "ppo.num_envs=4", "ppo.rollout_steps=16",
        "ppo.minibatch_size=64", "ppo.sequence_microbatch_size=64",
        f"ppo.total_steps={64 * (args.warmup_updates + args.measured_updates)}",
        "model.prediction_pretrain_steps=64", "experiment.eval_episodes=1",
        "model.prediction_interval_updates=0", "model.prediction_phase_updates=0",
        f"model.score_implementation={'autograd' if args.setting == 'autograd' else 'analytic'}",
        f"model.belief_compile={'inductor' if args.setting == 'inductor' else 'off'}",
        f"model.belief_cuda_graphs={'true' if args.setting == 'graphs' else 'false'}",
        f"experiment.output_dir={json.dumps(long_path(args.output / args.setting))}",
    ):
        command.extend(("--override", override))
    return compare(command)


if __name__ == "__main__":
    raise SystemExit(main())
