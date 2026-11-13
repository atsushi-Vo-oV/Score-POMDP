"""Analyze POPX campaigns without treating episodes as independent training seeds.

Usage: python scripts/popx_analyze.py results/campaigns/popx-* --out analysis/popx.md
Writes Markdown, tidy learning-curve CSV, paired-effect CSV and PNG curves.
"""
import argparse
import csv
from collections import defaultdict
import glob
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

LOO = {"no_f", "no_a2", "no_c", "no_d", "no_qd"}
SINGLE = {"full", "cem", "es", "pbt", "a2", "d"}
GRU_CONTRASTS = (("gru_full", "gru_ppo"), ("full", "gru_full"), ("ppo", "gru_ppo"))


def gru_contrast(row):
    return row["arm"].startswith("gru_") or row["reference"].startswith("gru_")


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def fmt(value):
    if not math.isfinite(value):
        return "NA"
    return f"{value:.3e}" if 0 < abs(value) < 1e-4 else f"{value:.4f}"


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else math.nan


def paired_interval(pairs, *, repeats=10000, seed=20261113):
    """Pairs = (checkpoint seed, continuation seed, difference).

    Frozen: resample checkpoint clusters, then continuation seeds within cluster.
    Scratch: ordinary paired training-seed bootstrap. Two checkpoints cannot give
    a well-calibrated population CI; the report states that limitation explicitly.
    """
    if not pairs:
        return math.nan, math.nan, math.nan
    values = np.asarray([row[2] for row in pairs])
    estimate = float(values.mean())
    if len(pairs) < 2:
        return estimate, math.nan, math.nan
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for ckpt, continuation, delta in pairs:
        groups[ckpt if ckpt >= 0 else continuation].append(delta)
    blocks = [np.asarray(v) for v in groups.values()]
    # Equal weight per independent training/checkpoint seed, including partial
    # campaigns with unequal numbers of completed continuation seeds.
    estimate = float(np.mean([block.mean() for block in blocks]))
    if len(blocks) < 2:
        return estimate, math.nan, math.nan
    samples = np.empty(repeats)
    for j in range(repeats):
        sample = [rng.choice(blocks[k], len(blocks[k]), replace=True).mean()
                  for k in rng.integers(len(blocks), size=len(blocks))]
        samples[j] = np.mean(sample)
    low, high = np.quantile(samples, [.025, .975])
    return estimate, float(low), float(high)


def discover(paths):
    roots = set()
    for expression in paths:
        for match in glob.glob(expression) or [expression]:
            p = Path(match)
            if p.name == "evaluation.csv" and p.is_file():
                roots.add(p.parent)
            elif p.exists():
                roots.update(q.parent for q in p.rglob("resolved_config.json") if q.parent.name.startswith("seed_"))
    runs = []
    identities = set()
    for root in sorted(roots):
        cfgpath = root / "resolved_config.json"
        if not cfgpath.is_file():
            continue
        config = json.loads(cfgpath.read_text(encoding="utf-8-sig"))
        if "popexplore" not in config:
            continue
        p = config["popexplore"]
        task, seed = root.parent.name, int(root.name.removeprefix("seed_"))
        identity = (p["group"], task, p["arm"], p["checkpoint_seed"], seed)
        if identity in identities:
            raise ValueError(f"duplicate experiment identity {identity}; select one campaign version")
        identities.add(identity)
        runs.append(dict(root=root, group=p["group"], task=task, arm=p["arm"], ckpt=p["checkpoint_seed"],
                         seed=seed, config=config, evaluation=read_csv(root/"evaluation.csv"),
                         members=read_csv(root/"population_evaluation.csv"), metrics=read_csv(root/"metrics.csv")))
    if not runs:
        raise ValueError("No population resolved_config.json files found")
    return runs


def learning_rows(runs):
    result = []
    for run in runs:
        members = defaultdict(list)
        for row in run["members"]:
            members[int(row["update"]), row["mode"]].append(row)
        for row in run["evaluation"]:
            update, mode = int(row["update"]), row["mode"]
            tested = members[update, mode]
            individual = [number(r,"mean_return") for r in tested if r["actor"].isdigit()]
            mean_actor = next((number(r,"mean_return") for r in tested if r["actor"] == "mean"), math.nan)
            f_on = run["config"]["popexplore"].get("optimizer_agent", {}).get("enabled", False)
            optimizer = next((number(r,"mean_return") for r in tested if r["actor"] == "optimizer"), math.nan)
            explorers = [number(r,"mean_return") for r in tested if r["actor"].isdigit() and (not f_on or r["actor"] != "0")]
            result.append(dict(group=run["group"], task=run["task"], arm=run["arm"], checkpoint_seed=run["ckpt"],
                               seed=run["seed"], mode=mode, update=update, global_step=int(row["global_step"]),
                               inference=number(row,"mean_return"), mean_actor=mean_actor,
                               best_individual=max(individual) if individual else math.nan,
                               population_mean=finite_mean(individual), optimizer=optimizer,
                               best_explorer=max(explorers) if explorers else math.nan))
    return result


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def effects(curves):
    groups = defaultdict(dict)
    for row in curves:
        key = row["group"], row["task"], row["mode"]
        armkey = row["arm"], row["checkpoint_seed"], row["seed"]
        groups[key].setdefault(armkey, {})[row["update"]] = row["inference"]
    output = []
    for (group, task, mode), arms in sorted(groups.items()):
        names = sorted({key[0] for key in arms})
        for arm in names:
            references_for_arm = list(("full",) if arm in LOO else ("ppo",) if arm in SINGLE else ())
            if any(name.startswith("gru_") for name in names):
                references_for_arm += [reference for name, reference in GRU_CONTRASTS if name == arm]
            for reference in references_for_arm:
                if arm == reference:
                    continue
                pairs, progress, references = [], [], []
                for (name, ckpt, seed), series in arms.items():
                    if name != arm or (reference, ckpt, seed) not in arms:
                        continue
                    other = arms[reference, ckpt, seed]
                    common = series.keys() & other.keys()
                    if common:
                        update = max(common)
                        pairs.append((ckpt, seed, series[update] - other[update]))
                        references.append(other[update])
                        progress.append(update)
                mean, low, high = paired_interval(pairs)
                output.append(dict(group=group, task=task, mode=mode, arm=arm, reference=reference,
                                   pairs=len(pairs), clusters=len({c if c >= 0 else s for c,s,_ in pairs}),
                                   updates=",".join(map(str, sorted(set(progress)))), difference=mean, ci_low=low, ci_high=high,
                                   paired_differences=json.dumps(pairs), reference_mean=finite_mean(references)))
    return output


def across_tasks(contrasts):
    groups = defaultdict(list)
    for r in contrasts:
        if r["pairs"]:
            groups[r["group"], r["mode"], r["arm"], r["reference"]].append(r)
    rows = []
    for (group, mode, arm, reference), tasks in sorted(groups.items()):
        tables = [{(c,s):d / max(1., abs(r["reference_mean"])) for c,s,d in json.loads(r["paired_differences"])} for r in tasks]
        shared = set.intersection(*(set(t) for t in tables))
        pairs = [(c,s,float(np.mean([t[c,s] for t in tables]))) for c,s in sorted(shared)]
        mean, lo, hi = paired_interval(pairs)
        rows.append(dict(group=group, mode=mode, arm=arm, reference=reference,
                         tasks=",".join(r["task"] for r in tasks), task_count=len(tasks), pairs=len(pairs),
                         difference=mean, ci_low=lo, ci_high=hi, paired_differences=json.dumps(pairs)))
    return rows


def optimizer_events(runs):
    rows = []
    for run in runs:
        if not run["config"]["popexplore"].get("optimizer_agent", {}).get("enabled", False):
            continue
        previous = None
        for r in run["metrics"]:
            if not math.isfinite(number(r, "optimizer_candidate")):
                continue
            gain = number(r,"optimizer_return_after") - number(r,"optimizer_return_before")
            previous_gain = (number(previous,"optimizer_return_after") - number(previous,"optimizer_return_before")) if previous else math.nan
            undo = bool(previous and number(previous,"optimizer_adopted") == 1 and previous_gain > 0
                        and number(r,"optimizer_ppo_return_change") <= -previous_gain)
            rows.append(dict(group=run["group"], task=run["task"], arm=run["arm"], seed=run["seed"], update=int(r["update"]),
                             **{k:number(r,k) for k in r if k.startswith("optimizer_")}, adoption_gain=gain, ppo_undo=int(undo)))
            previous = r
    return rows


def diagnose(runs, curves, contrasts):
    rows = []
    for run in runs:
        label = f"{run['group']}/{run['task']}/{run['arm']}/c{run['ckpt']}/s{run['seed']}"
        measured = [r for r in run["metrics"] if math.isfinite(number(r,"fitness_return_std"))]
        noise = finite_mean([number(r,"fitness_noise_se") for r in measured])
        spread = finite_mean([number(r,"fitness_return_std") for r in measured])
        corr = finite_mean([number(r,"rank_stability") for r in measured])
        status = "要注意" if math.isfinite(noise) and noise >= spread else ("非該当" if math.isfinite(noise) and math.isfinite(spread) else "判定不可")
        rows.append((label, "個体差より評価ノイズが大きい", status,
                     "2反復平均差のRMS/√2 ≥ 個体平均収益の標準偏差", f"noise={fmt(noise)}, spread={fmt(spread)}, rankρ={fmt(corr)}"))
        changed = [r for r in measured if number(r,"selection_replaced") > 0]
        loss = finite_mean([number(r,"selection_after_return")-number(r,"selection_before_return") for r in changed])
        status = "要注意" if math.isfinite(loss) and loss < -noise else ("非該当" if math.isfinite(loss) and math.isfinite(noise) else "判定不可")
        rows.append((label, "選抜・再標本化で収益が失われる", status,
                     "同一評価seedの選抜後−前 < −fitness_noise_se", f"after−before={fmt(loss)}"))
        series = sorted([r for r in curves if (r["group"],r["task"],r["arm"],r["checkpoint_seed"],r["seed"],r["mode"]) ==
                         (run["group"],run["task"],run["arm"],run["ckpt"],run["seed"],"mean_chain")], key=lambda r:r["update"])
        if run["config"]["popexplore"]["population_size"] == 1:
            status, evidence = "判定不可", "単独actorのため平均化なし"
        elif len(series) >= 2:
            first, last = series[0], series[-1]
            gain = last["population_mean"] - first["population_mean"]
            gap = last["population_mean"] - last["mean_actor"]
            threshold = max(.1, .1 * abs(last["population_mean"]))
            status = "要注意" if gain > 0 and gap > threshold else "非該当"
            evidence = f"population gain={fmt(gain)}, population−mean actor={fmt(gap)}"
        else:
            status, evidence = "判定不可", "評価点が2点未満"
        rows.append((label, "集団は改善するが平均actorが悪い", status,
                     "集団平均の初回比>0、集団平均−平均actor>max(0.1,|集団平均|×0.1)", evidence))
        ratio = finite_mean([number(r,"rnd_external_ratio") for r in run["metrics"]])
        rows.append((label, "内発報酬が外部報酬を上回る", "要注意" if ratio > 1 else "非該当",
                     "平均 β×RND / 平均 |外部報酬| > 1", f"ratio={fmt(ratio)}"))
        if run["config"]["popexplore"].get("optimizer_agent", {}).get("enabled", False):
            f = run["config"]["popexplore"]["optimizer_agent"]
            events = optimizer_events([run])
            blocked = [r for r in events if number(r,"optimizer_explorer_best") - number(r,"optimizer_fitness_return") > f["margin"]
                       and number(r,"optimizer_difference") > f["margin"] and number(r,"optimizer_adopted") == 0]
            rows.append((label, "探索側は良い個体を見つけたが取り込まれない（判定が厳しすぎる候補）",
                         "設定で無効" if f["adopt"] == "none" else "要注意" if blocked else "非該当" if events else "判定不可",
                         "探索最良−F > margin、対応差平均 > margin、採用なし（window不足/LCBを確認）",
                         f"該当更新={len(blocked)}; 最新pairs={fmt(number(events[-1],'optimizer_pair_count')) if events else 'NA'}"))
            count = sum(r["ppo_undo"] for r in events)
            adopted = [r for r in events if number(r,"optimizer_adopted") == 1]
            rows.append((label, "取り込んだ直後にPPOが元へ戻す", "要注意" if count else "非該当" if adopted else "判定不可",
                         "直前PPO更新で採用し即時改善>0、今回の同一seedでのPPO前後差 ≤ −その改善量",
                         f"巻き戻り候補={count}; 採用={len(adopted)}（異なるupdateの差量比較は探索的）"))
    return rows


def gru_section(runs, curves, contrasts, events):
    selected = [r for r in runs if r["arm"].startswith("gru_")]
    if not selected:
        return []
    lines = ["", "## GRU 比較", "",
             "gru_full − gru_ppo はGRU内の集団探索効果、full − gru_full と ppo − gru_ppo は同じ学習予算でのlightとの比較。",
             "対応seedごとの最新共通評価updateを使用し、未完走を含む。共通評価がなければ差・区間はNA。",
             "粒子数等のGRUで使わないキーは計算量の一致を意味しない。実行設定とFLOPsの比較は別途確認する。", "",
             f"GRU読込: {len(selected)} runs、評価未到達: {sum(not r['evaluation'] for r in selected)} runs。", "",
             "|群/課題|mode|腕−参照|対応seed数|共通update|平均差|95%区間|seedごとの差|",
             "|---|---|---|---:|---|---:|---|---|"]
    for r in contrasts:
        if gru_contrast(r):
            individual = "; ".join(f"{seed}: {fmt(delta)}" for _, seed, delta in json.loads(r["paired_differences"])) or "NA"
            lines.append(f"|{r['group']}/{r['task']}|{r['mode']}|{r['arm']}−{r['reference']}|{r['pairs']}|{r['updates'] or 'NA'}|{fmt(r['difference'])}|[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]|{individual}|")
    lines += ["", "### GRU 最新成績", "", "|課題/腕/seed|mode|update|推論収益|F収益|探索最良|", "|---|---|---:|---:|---:|---:|"]
    latest = {}
    for row in curves:
        if row["arm"].startswith("gru_"):
            key = tuple(row[k] for k in ("group", "task", "arm", "checkpoint_seed", "seed", "mode"))
            if key not in latest or row["update"] > latest[key]["update"]:
                latest[key] = row
    for _, r in sorted(latest.items()):
        lines.append(f"|{r['task']}/{r['arm']}/{r['seed']}|{r['mode']}|{r['update']}|{fmt(r['inference'])}|{fmt(r['optimizer'])}|{fmt(r['best_explorer'])}|")
    lines += ["", "### gru_full の F 診断", "",
              "採用イベントと各runの最新判定を表示。採用前後と次PPO前後は測定したupdateだけ値を持つ。未測定はNA。",
              "|課題/seed|update|累積採用|採用|候補|pairs|対応差|下側限界|fitness F|探索最良|採用前|採用後|PPO前後差|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    f_events = [r for r in events if r["arm"] == "gru_full"]
    latest_events = {(r["group"], r["task"], r["seed"]): r for r in f_events}
    for r in f_events:
        if r["optimizer_adopted"] == 1 or r is latest_events[r["group"], r["task"], r["seed"]]:
            fields = ("optimizer_adoptions", "optimizer_adopted", "optimizer_candidate", "optimizer_pair_count",
                      "optimizer_difference", "optimizer_lower_bound", "optimizer_fitness_return",
                      "optimizer_explorer_best", "optimizer_return_before", "optimizer_return_after", "optimizer_ppo_return_change")
            lines.append(f"|{r['task']}/{r['seed']}|{r['update']}|" + "|".join(fmt(number(r, k)) for k in fields) + "|")
    lines += ["", "|run|診断|判定|規則|観測値|", "|---|---|---|---|---|"]
    for row in diagnose(selected, curves, contrasts):
        lines.append("|" + "|".join(str(x).replace("|", "\\|") for x in row) + "|")
    return lines


def analyze(paths, out, plots=True):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = discover(paths)
    curves = learning_rows(runs)
    contrasts = effects(curves)
    cross = across_tasks(contrasts)
    events = optimizer_events(runs)
    curvepath = out.with_name(out.stem + "_curves.csv")
    write_csv(curvepath, curves, list(curves[0]) if curves else ["group","task","arm","seed","mode","update","inference"])
    effectpath = out.with_name(out.stem + "_effects.csv")
    effectfields = list(contrasts[0]) if contrasts else ["group", "task", "mode", "arm", "reference", "pairs", "difference", "ci_low", "ci_high"]
    write_csv(effectpath, contrasts, effectfields)
    crosspath = out.with_name(out.stem + "_across_tasks.csv")
    write_csv(crosspath, cross, list(cross[0]) if cross else ["group","arm","reference","pairs","difference"])
    eventpath = out.with_name(out.stem + "_optimizer_events.csv")
    write_csv(eventpath, events, list(events[0]) if events else ["task","arm","seed","update","optimizer_adopted"])
    lines = ["# POPX 寄与解析", "", f"読込: {len(runs)} runs。外部エピソード収益を使用。", "",
             "差は『腕 − 参照腕』。leave-one-out の負差は除いた部品の有用性と整合するが、相互作用と学習ノイズを含む。",
             "12腕: full/no_f/no_a2/no_c/no_d/no_qd/cem/es/pbt/a2/d/ppo。5つの除去腕はfull、単独腕とfullはppoを参照する。",
             "95%区間は対応する学習seedのbootstrap（10000反復）。5 seedでも探索的区間であり、多重比較補正した有意性を主張しない。",
             "1独立cluster以下は区間NA。未完了run間の差は各対応ペアの最新共通評価updateを使う。評価episodeを学習seedの代用にしない。",
             "最良個体は評価集合での最大値（選択バイアスあり）。推論bestは別のfitness評価で選んだ個体。集団平均は個体収益の平均、平均actorは重み平均/CEM μの収益。", "",
             f"評価未到達: {sum(not r['evaluation'] for r in runs)} runs（欠損を0で補わない）。最新値は途中経過を含む。", "",
             "## 最新成績", "", "|群/課題/腕|mode|runs|最新update範囲|推論actor|最良個体|集団平均収益|分布平均actor|最適化専用|探索最良|", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    latest = {}
    for row in curves:
        key = tuple(row[k] for k in ("group","task","arm","checkpoint_seed","seed","mode"))
        if key not in latest or row["update"] > latest[key]["update"]:
            latest[key] = row
    summary = defaultdict(list)
    for row in latest.values():
        summary[row["group"],row["task"],row["arm"],row["mode"]].append(row)
    for (group, task, arm, mode), values in sorted(summary.items()):
        if arm.startswith("gru_"):
            continue
        fields = [fmt(finite_mean([r[k] for r in values])) for k in ("inference","best_individual","population_mean","mean_actor","optimizer","best_explorer")]
        lines.append(f"|{group}/{task}/{arm}|{mode}|{len(values)}|{min(r['update'] for r in values)}–{max(r['update'] for r in values)}|"+"|".join(fields)+"|")
    lines += ["", "## 対応差と区間", "", "|群/課題|mode|腕−参照|pairs/clusters|共通update|平均差|95%区間|", "|---|---|---|---:|---|---:|---|"]
    for r in contrasts:
        if gru_contrast(r):
            continue
        lines.append(f"|{r['group']}/{r['task']}|{r['mode']}|{r['arm']}−{r['reference']}|{r['pairs']}/{r['clusters']}|{r['updates']}|{fmt(r['difference'])}|[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]|")
    lines += ["", "## 課題横断の要約", "",
              "課題ごとに Δ/max(1,|対応参照腕の平均収益|) として規模を揃え、課題を等重みで平均する。単位の異なる生収益は合算しない。",
              "当該対比の利用可能な課題すべてに共通するseedだけを使い、同じseedの課題ベクトルを一緒にbootstrapする。課題が1つなら課題横断の証拠にはならない。",
              "|群/mode|対比|課題|共通seed数|正規化差|95%区間|", "|---|---|---|---:|---:|---|"]
    for r in cross:
        if gru_contrast(r):
            continue
        lines.append(f"|{r['group']}/{r['mode']}|{r['arm']}−{r['reference']}|{r['tasks']}|{r['pairs']}|{fmt(r['difference'])}|[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]|")
    lines += ["", "## 最適化専用個体の取り込み", "",
              "前/後は採用イベントにおける同一評価seedの値。PPO前/後も各update内で同一seed。学習rollout収益とは区別する。",
              "|課題/腕/seed|update|累積採用|採用|候補|対応差|下側限界|採用前|採用後|PPO前後差|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    latest_events = {}
    for r in events:
        latest_events[r["group"],r["task"],r["arm"],r["seed"]] = r
    for r in events:
        if r["arm"].startswith("gru_"):
            continue
        if r["optimizer_adopted"] == 1 or r is latest_events[r["group"],r["task"],r["arm"],r["seed"]]:
            values = [fmt(r[k]) for k in ("optimizer_adoptions","optimizer_adopted","optimizer_candidate","optimizer_difference","optimizer_lower_bound","optimizer_return_before","optimizer_return_after","optimizer_ppo_return_change")]
            lines.append(f"|{r['task']}/{r['arm']}/{r['seed']}|{r['update']}|" + "|".join(values) + "|")
    lines += ["", f"全F診断時系列: [{eventpath.name}]({eventpath.name})。課題横断個票: [{crosspath.name}]({crosspath.name})。"]
    lines += ["", "## 失敗時の切り分け（判定規則付き）", "", "|run|診断|判定|規則|観測値|", "|---|---|---|---|---|"]
    for row in diagnose([r for r in runs if not r["arm"].startswith("gru_")], curves, contrasts):
        lines.append("|"+"|".join(str(x).replace("|","\\|") for x in row)+"|")
    lines += gru_section(runs, curves, contrasts, events)
    lines += ["", "## 学習曲線", "", f"全測定点: [{curvepath.name}]({curvepath.name})。対応差の個票: [{effectpath.name}]({effectpath.name})。"]
    if plots and importlib.util.find_spec("matplotlib") is None:
        plots = False
        lines += ["", "matplotlibがない環境のため、曲線の各点をCSVと次の表で出力した（追加インストールなし）。"]
    if not plots:
        points = defaultdict(list)
        for row in curves:
            points[row["group"],row["task"],row["arm"],row["mode"],row["update"]].append(row)
        lines += ["", "|群/課題/腕|mode|update|推論actor|最良個体|集団平均収益|", "|---|---|---:|---:|---:|---:|"]
        for (group,task,arm,mode,update), values in sorted(points.items()):
            vals = [fmt(finite_mean([r[k] for r in values])) for k in ("inference","best_individual","population_mean")]
            lines.append(f"|{group}/{task}/{arm}|{mode}|{update}|"+"|".join(vals)+"|")
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        grouped = defaultdict(list)
        for row in curves:
            grouped[row["group"],row["task"],row["mode"]].append(row)
        for (group, task, mode), values in sorted(grouped.items()):
            fig, axes = plt.subplots(1, 3, figsize=(14,4), sharex=True)
            for axis, column in zip(axes, ("inference","best_individual","population_mean")):
                for arm in sorted({r["arm"] for r in values}):
                    points = defaultdict(list)
                    for r in values:
                        if r["arm"] == arm:
                            points[r["update"]].append(r[column])
                    x = sorted(points)
                    axis.plot(x, [finite_mean(points[k]) for k in x], label=arm)
                axis.set(title=column, xlabel="update", ylabel="external episode return")
                axis.grid(alpha=.2)
            axes[-1].legend(fontsize=7)
            fig.suptitle(f"{group} / {task} / {mode}")
            fig.tight_layout()
            path = out.with_name(f"{out.stem}_{group}_{task}_{mode}.png")
            fig.savefig(path, dpi=130)
            plt.close(fig)
            lines += ["", f"![{group}/{task}/{mode}]({path.name})"]
    out.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"runs":len(runs), "curve_rows":len(curves), "contrasts":len(contrasts), "out":str(out)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaigns", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    print(json.dumps(analyze(args.campaigns, args.out, plots=not args.no_plots), ensure_ascii=False))


if __name__ == "__main__":
    main()
