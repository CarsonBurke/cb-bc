"""Generate charts for the tensor-bot postmortem PDF.

Reads per-session train_log.jsonl files and produces:
  - timeline.png         : Gantt-style timeline of notable innovations.
  - flagship_runs.png    : episode-resolved winrate / reward / EV curves
    for representative flagship runs from the project's lifetime.
  - throughput.png       : per-episode environment-step volume across
    architecture inflection points.
  - pretrain_curves.png  : BC actor loss + critic-pretrain explained variance.
  - architecture.png     : block diagram of actor + segmenting critic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path("/home/marvin/Documents/repositories/cb-bc")
SESSIONS = ROOT / "bots" / "tensor" / "sessions"
OUT = ROOT / "postmortem"
OUT.mkdir(exist_ok=True)

DATE_RE = re.compile(r"(\d{8})")


def parse_jsonl(path: Path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def session_date(name: str):
    m = DATE_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except Exception:
        return None


# Notable innovations grouped into Gantt-style swim lanes.
# (label, start, end, lane, color) — bands span the period when each idea was
# the dominant change, anchored to the actual git commit dates.
LANES = [
    "Actor backbone",
    "Action space",
    "Critic / value",
    "Training pipeline",
    "Reward / credit",
    "Pretraining",
]

INNOV_BANDS = [
    # Actor backbone — the deployable inference network.
    # The minGRU recurrent state (H_DIM=24) is constant from 03-21 to project
    # end; it is not a "phase" and is omitted as a band.
    ("MLP trunk",                           "2026-03-16", "2026-03-22", 0, "#7aa6c2"),
    ("Shared per-tile encoder",             "2026-03-21", "2026-04-04", 0, "#8dbedb"),
    ("CLS-attention transformer",           "2026-04-05", "2026-04-09", 0, "#a3d2eb"),
    ("RBF tile SA + peri-LN",               "2026-04-10", "2026-04-15", 0, "#b8def0"),
    ("SuSIE + look-away + XSA",             "2026-04-16", "2026-04-20", 0, "#cfe9f6"),

    # Action space — sampling, exploration, and the per-head policy controls.
    ("Per-head logit scale",                "2026-03-23", "2026-03-25", 1, "#cda36b"),
    ("DreamerV3 unimix exploration",        "2026-03-26", "2026-04-04", 1, "#dfb87f"),
    ("State-dependent per-head temperature", "2026-04-05", "2026-04-20", 1, "#eccea0"),
    ("Logit softcap + temp init",           "2026-04-12", "2026-04-20", 1, "#f5dfb9"),
    ("Adaptive per-head entropy",           "2026-04-07", "2026-04-20", 1, "#f9eccd"),

    # Critic / value
    ("MSE critic",                          "2026-03-16", "2026-03-22", 2, "#bf7575"),
    ("DreamerV3 symlog twohot",             "2026-03-23", "2026-03-31", 2, "#cf8c8c"),
    ("Transformer critic + HL-Gauss",       "2026-04-03", "2026-04-15", 2, "#dfa3a3"),
    ("Cross-vis SDPA segmenting critic",    "2026-04-16", "2026-04-20", 2, "#efbaba"),

    # Training pipeline
    ("Scripted-dummy curriculum",           "2026-03-16", "2026-03-19", 3, "#79a37a"),
    ("Batched GPU trainer + CUDA graphs",   "2026-03-20", "2026-04-01", 3, "#8db98e"),
    ("Shared-mem ring + strict schema",     "2026-04-02", "2026-04-11", 3, "#a1cfa3"),
    ("NorMuon + PMPO + H2D prefetch",       "2026-04-12", "2026-04-20", 3, "#b6e5b8"),

    # Reward / credit
    ("Hand-shaped placement deltas",        "2026-03-16", "2026-03-21", 4, "#c2a16f"),
    ("Conv flow rewards + conv_dir head",   "2026-03-19", "2026-03-29", 4, "#d2b384"),
    ("Per-team flow + delivery rewards",    "2026-04-01", "2026-04-20", 4, "#f1d8b0"),

    # Pretraining — actor BC and critic HL-Gauss pretrain are separate stages.
    ("Actor BC on >=1500 ELO replays",      "2026-03-26", "2026-04-09", 5, "#9a86bd"),
    ("Encoder pretrain",                    "2026-04-06", "2026-04-09", 5, "#ae9bcc"),
    ("BC actor >=2150 ELO",                 "2026-04-19", "2026-04-20", 5, "#c2b1da"),
    ("Critic HL-Gauss pretrain",            "2026-04-19", "2026-04-20", 5, "#d4c5e3"),
]


def make_timeline(out_path):
    """Lay out lane bands and decide which labels render in-band vs as
    external callouts, using actual matplotlib text-width measurements
    against each band's pixel width so labels never visually overflow.
    """
    bands = []
    for label, start_str, end_str, lane, color in INNOV_BANDS:
        x0 = datetime.strptime(start_str, "%Y-%m-%d").date().toordinal()
        x1 = datetime.strptime(end_str, "%Y-%m-%d").date().toordinal()
        bands.append({"label": label, "x0": x0, "x1": x1, "lane": lane, "color": color})

    # Greedy interval coloring within each lane.
    by_lane = {}
    for b in bands:
        by_lane.setdefault(b["lane"], []).append(b)
    for lane, items in by_lane.items():
        items.sort(key=lambda b: b["x0"])
        rows = []
        for b in items:
            placed = False
            for ri, last_end in enumerate(rows):
                if b["x0"] >= last_end:
                    rows[ri] = b["x1"]
                    b["row"] = ri
                    placed = True
                    break
            if not placed:
                b["row"] = len(rows)
                rows.append(b["x1"])
        b_max = max(b["row"] for b in items)
        for b in items:
            b["lane_n_rows"] = b_max + 1

    FIXED_FONT = 6.5
    PADDING_PX = 8  # left + right padding inside band

    start = date(2026, 3, 16).toordinal()
    end = date(2026, 4, 22).toordinal()
    right_buffer = 8

    # Create the figure up-front so text-width measurements use the same
    # data-to-pixel transform we will render with. Horizontal layout is
    # determined by figsize_w and xlim, both fixed here; the height is
    # adjusted later once we know how many rows the lanes need.
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=140)
    ax.set_xlim(start - 1, end + right_buffer)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def text_w_px(text):
        t = ax.text(0, 0, text, fontsize=FIXED_FONT)
        bbox = t.get_window_extent(renderer)
        t.remove()
        return bbox.width

    def band_w_px(x0, x1):
        return ax.transData.transform((x1, 0))[0] - ax.transData.transform((x0, 0))[0]

    def fit_band(label, x0, x1):
        bw = band_w_px(x0, x1) - PADDING_PX
        if bw <= 0:
            return [label], False
        if text_w_px(label) <= bw:
            return [label], True
        words = label.split()
        if len(words) < 2:
            return [label], False
        best = None
        for i in range(1, len(words)):
            l1, l2 = " ".join(words[:i]), " ".join(words[i:])
            w1, w2 = text_w_px(l1), text_w_px(l2)
            if w1 <= bw and w2 <= bw:
                score = max(w1, w2)
                if best is None or score < best[0]:
                    best = (score, [l1, l2])
        if best is not None:
            return best[1], True
        return [label], False

    for b in bands:
        lines, fits = fit_band(b["label"], b["x0"], b["x1"])
        b["fontsize"] = FIXED_FONT
        b["external"] = not fits
        b["lines"] = lines

    # Per-band row height. In-band bands need vertical room for their wrapped
    # lines; external-label bands need a label zone above plus a colored
    # stripe below (same total height as a 2-line inline band).
    line_h = 0.40
    base_row_h = 0.75
    lane_pad = 0.45
    for b in bands:
        if b["external"]:
            b["height"] = max(base_row_h, 2 * line_h + 0.25)
        else:
            n = len(b["lines"])
            b["height"] = max(base_row_h, n * line_h + 0.25)

    lane_starts = {}
    cur_y = 0.0
    lane_row_offsets = {}
    for lane in range(len(LANES)):
        items = by_lane.get(lane, [])
        lane_starts[lane] = cur_y
        n_rows = items[0]["lane_n_rows"] if items else 1
        row_heights = []
        for r in range(n_rows):
            in_row = [b for b in items if b["row"] == r]
            row_heights.append(max((b["height"] for b in in_row), default=base_row_h))
        offsets = []
        y = 0.0
        for h in row_heights:
            offsets.append(y)
            y += h
        lane_row_offsets[lane] = (offsets, row_heights)
        cur_y += y + lane_pad
    total_h = cur_y

    fig_h = max(5.0, 0.75 * total_h + 1.2)
    fig.set_size_inches(11.5, fig_h)

    for b in bands:
        offsets, row_heights = lane_row_offsets[b["lane"]]
        y = lane_starts[b["lane"]] + offsets[b["row"]]
        h = row_heights[b["row"]] * 0.88
        if b["external"]:
            # Top portion = label area (label may be wider than the band's
            # x-extent and overflow horizontally, but it lives in a vertical
            # zone above the colored stripe so it cannot collide with text in
            # neighboring in-band cells of the same row).
            label_zone = h * 0.42
            gap = h * 0.05
            band_h = h - label_zone - gap
            band_y = y + label_zone + gap
            ax.add_patch(mpatches.FancyBboxPatch(
                (b["x0"], band_y), b["x1"] - b["x0"], band_h,
                boxstyle="round,pad=0.02,rounding_size=0.4",
                linewidth=0.6, edgecolor="#333", facecolor=b["color"], alpha=0.93,
            ))
            text_x = b["x0"] + (b["x1"] - b["x0"]) / 2
            text_y = y + label_zone / 2
            ax.text(text_x, text_y, b["label"],
                    ha="center", va="center",
                    fontsize=FIXED_FONT, color="#111")
        else:
            ax.add_patch(mpatches.FancyBboxPatch(
                (b["x0"], y), b["x1"] - b["x0"], h,
                boxstyle="round,pad=0.02,rounding_size=0.4",
                linewidth=0.6, edgecolor="#333", facecolor=b["color"], alpha=0.93,
            ))
            text_x = b["x0"] + (b["x1"] - b["x0"]) / 2
            cy = y + h / 2
            ax.text(text_x, cy, "\n".join(b["lines"]),
                    ha="center", va="center",
                    fontsize=FIXED_FONT, color="#111",
                    linespacing=1.05)

    yticks, yticklabels = [], []
    for lane in range(len(LANES)):
        offsets, row_heights = lane_row_offsets[lane]
        center = lane_starts[lane] + sum(row_heights) / 2
        yticks.append(center)
        yticklabels.append(LANES[lane])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9, fontweight="bold")

    for lane in range(1, len(LANES)):
        ax.axhline(lane_starts[lane] - lane_pad / 2,
                   color="#bbb", linewidth=0.5, alpha=0.7)

    ticks = list(range(start, end + 1, 4))
    ax.set_xticks(ticks)
    ax.set_xticklabels([datetime.fromordinal(t).strftime("%m-%d") for t in ticks],
                       fontsize=8)
    ax.set_ylim(-0.3, total_h + 0.1)
    ax.invert_yaxis()
    ax.set_xlabel("Date (2026)")
    ax.set_title("Project timeline of notable innovations: actor backbone, action space, critic, training, rewards, pretraining")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.4)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# Flagship runs spanning the full transformer-era project history. Each run
# is from a distinct architectural regime; the early MLP-only runs were
# dropped because their resources-per-episode is ~1 (different reward scale).
FLAGSHIP_RUNS = [
    # (session_name, label, color)
    ("sp_outcome_20260330",                "Outcome-only training (03-30)", "#7a9bd1"),
    ("sp_income_20260330c",                "Income shaping + flow (03-30)", "#406a9c"),
    ("sp_20260401",                        "BFS flow distance + KL-LR (04-01)", "#1e3a64"),
    ("horizon_50x1_dummy500_20260404",     "Horizon-50 dummy curriculum (04-04)", "#7fa564"),
    ("sp_20260405_193453",                 "CLS-attention actor (04-05)",       "#a23c2b"),
    ("sp_encoder_pt_20260407_004234",      "Encoder pretrain + adaptive ent (04-07)", "#c08a2b"),
    ("sp_20260409_manual36",               "RBF tile SA + peri-LN (04-09)",      "#5a3286"),
    ("sp_20260409_manual39",               "NorMuon + PMPO + softcap (04-09)",   "#2f6f4a"),
    ("sp_20260417_115120",                 "Cross-visibility SDPA critic (04-17)", "#9c2666"),
]


def smooth(xs, ys, window=15):
    if len(ys) < window:
        return xs, ys
    out = []
    n = len(ys)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return xs, out


def make_flagship_runs(out_path):
    """Per-session rolling mean of resources harvested per episode."""
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=140)
    plotted = 0
    for sess, label, color in FLAGSHIP_RUNS:
        log = SESSIONS / sess / "train_log.jsonl"
        if not log.exists():
            continue
        rows = [r for r in parse_jsonl(log) if "episode" in r and "resources" in r]
        if not rows:
            continue
        xs = [r["episode"] for r in rows]
        ys = [float(r["resources"]) for r in rows]
        # Drop sessions that emit only zeros (some flow-only runs report 0
        # because their reward channel is `r_flow`, not `resources`).
        if max(ys) <= 0.01:
            continue
        xs, ys = smooth(xs, ys, window=20)
        ax.plot(xs, ys, label=label, color=color, linewidth=1.8, alpha=0.92)
        plotted += 1

    ax.set_ylabel("Mean cumulative harvested resources / episode\n(20-ep rolling mean, symlog)")
    ax.set_xlabel("Training episode within run")
    ax.set_title("Mean cumulative harvested resources — flagship training runs across project history")
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.0, ncol=2, framealpha=0.95)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(bottom=0, top=600)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def collect_session_summary():
    out = []
    for d in SESSIONS.iterdir():
        if not d.is_dir():
            continue
        log = d / "train_log.jsonl"
        if not log.exists():
            continue
        sd = session_date(d.name)
        if sd is None:
            continue
        rows = parse_jsonl(log)
        rows = [r for r in rows if "episode" in r]
        if not rows:
            continue
        sps_vals = [r.get("sps") for r in rows if isinstance(r.get("sps"), (int, float))]
        max_sps = max(sps_vals) if sps_vals else 0
        out.append({"name": d.name, "date": sd, "max_sps": max_sps})
    return out


def make_throughput(summary, out_path):
    by_date = {}
    for s in summary:
        if s["max_sps"] > 0:
            by_date.setdefault(s["date"], []).append(s["max_sps"])
    points = sorted([(d, max(v)) for d, v in by_date.items()])
    xs = [p[0].toordinal() for p in points]
    ys = [p[1] / 1000.0 for p in points]

    fig, ax = plt.subplots(figsize=(11.5, 3.6), dpi=140)
    ax.bar(xs, ys, color="#5a8", alpha=0.78, width=0.85)

    regimes = [
        ("2026-03-18", "MLP self-play\n~195K steps/ep"),
        ("2026-03-23", "Distributional critic\n~95K"),
        ("2026-04-05", "CLS-attention actor\n~32K"),
        ("2026-04-10", "RBF + transformer critic\n~27K"),
    ]
    for d, label in regimes:
        x = datetime.strptime(d, "%Y-%m-%d").date().toordinal()
        ax.axvline(x, color="#444", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.text(x, max(ys) * 0.95, label, fontsize=7, ha="center", va="top",
                bbox=dict(facecolor="white", edgecolor="#aaa", boxstyle="round,pad=0.2", alpha=0.85))

    all_x = sorted(set(xs))
    tick_x = all_x[:: max(1, len(all_x) // 10)]
    ax.set_xticks(tick_x)
    ax.set_xticklabels([datetime.fromordinal(t).strftime("%m-%d") for t in tick_x], fontsize=8)
    ax.set_xlabel("Date (2026)")
    ax.set_ylabel("Step volume per episode (k steps)")
    ax.set_title("Per-episode env-step volume vs. architecture inflection points")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_pretrain_curves(out_path):
    bc_path = SESSIONS / "bc_actor_20260420_135121" / "train_log.jsonl"
    cr_path = SESSIONS / "critic_pretrain_income_20260420_145944" / "train_log.jsonl"

    bc_rows = [r for r in parse_jsonl(bc_path) if "loss" in r]
    cr_rows = [r for r in parse_jsonl(cr_path) if "explained_var" in r]

    bc_x = [r.get("episode", i) for i, r in enumerate(bc_rows)]
    bc_y = [r["loss"] for r in bc_rows]

    head_specs = [
        ("loss_bt",  "build_type CE",     "#3a7d44"),
        ("loss_bd",  "build_dir CE",      "#1f4f8b"),
        ("loss_cd",  "conv_dir CE",       "#a23c2b"),
        ("loss_brt", "bridge_target CE",  "#b16f1e"),
        ("loss_md",  "move_dir CE",       "#5a3286"),
    ]
    bc_heads = {
        key: [r.get(key, 0.0) for r in bc_rows]
        for key, _label, _color in head_specs
    }

    cr_x = [r.get("episode", i) for i, r in enumerate(cr_rows)]
    cr_y = [r["explained_var"] for r in cr_rows]
    cr_loss = [r.get("loss", r.get("value_loss", 0.0)) for r in cr_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=140)
    ax = axes[0]
    ax.plot(bc_x, bc_y, label="total CE loss", color="#000", linewidth=1.6)
    for key, label, color in head_specs:
        ax.plot(bc_x, bc_heads[key], label=label, color=color,
                linewidth=1.0, alpha=0.85)
    ax.set_title("Actor BC pretrain (>=2150 ELO replays)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cross-entropy loss (symlog)")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.5)
    # Place the legend below the axes so it never overlaps the curves.
    ax.legend(fontsize=7.6, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False)

    ax = axes[1]
    ev_line, = ax.plot(cr_x, cr_y, color="#225", linewidth=1.6,
                       label="explained variance")
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Explained variance", color="#225")
    ax.tick_params(axis="y", labelcolor="#225")
    ax.grid(linestyle="--", linewidth=0.4, alpha=0.5)
    ax2 = ax.twinx()
    loss_line, = ax2.plot(cr_x, cr_loss, color="#a44", alpha=0.65,
                          linewidth=0.9, label="HL-Gauss value loss")
    ax2.set_ylabel("HL-Gauss value loss", color="#a44")
    ax2.tick_params(axis="y", labelcolor="#a44")
    ax.set_title("Centralized critic pretrain (income returns, HL-Gauss)")
    # Combined legend below the axes, matching the actor BC subplot.
    ax.legend(
        handles=[ev_line, loss_line], fontsize=7.6, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_architecture(out_path):
    fig, ax = plt.subplots(figsize=(11.5, 7.1), dpi=150)
    ax.axis("off")
    ax.set_xlim(0, 110)
    ax.set_ylim(-2, 78)

    def box(x, y, w, h, label, fc="#dfeefb", ec="#225", fs=8.5, lw=1.0):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4,rounding_size=0.6",
            linewidth=lw, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)

    def stacked_box(x, y, w, h, label, n_layers=3, fc="#dfeefb", ec="#225", fs=8.5,
                    offset=0.6):
        """Draw a stacked-rectangle visualization (n_layers boxes, offset diagonally),
        with the label rendered on the front-most box."""
        for i in range(n_layers - 1, -1, -1):
            ox = i * offset
            oy = -i * offset
            face = fc
            edge = ec
            # Tint the back layers slightly darker for depth.
            if i > 0:
                face = _shade(fc, 1 - 0.12 * i)
            rect = mpatches.FancyBboxPatch(
                (x + ox, y + oy), w, h,
                boxstyle="round,pad=0.4,rounding_size=0.6",
                linewidth=1.0, edgecolor=edge, facecolor=face,
            )
            ax.add_patch(rect)
        # Label on the front-most layer (i=0).
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, color="#225"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0, shrinkA=2, shrinkB=2))

    ax.text(55, 75, "Actor (stdlib-deployable) and per-unit segmenting critic",
            ha="center", va="center", fontsize=12, fontweight="bold")

    # ---- ACTOR (left column) ----
    ax.text(25, 70, "Actor — model.py:2842 (BuilderMLP)",
            ha="center", fontsize=9.5, fontweight="bold")
    box(2, 58, 46, 8,
        "obs[545] = 5x5 tile feats (525) + globals (20)\n+ recurrent h_prev (24)",
        fc="#fafafa")
    box(2, 48, 14, 6, "tile_linear\n(23->8)")
    box(18, 48, 14, 6, "tile self-attn\nRBF, SuSIE pos,\nlook-away reg, XSA")
    box(34, 48, 14, 6, "per-tile FFN\nLeakyReLU^2")
    box(10, 38, 30, 6, "CLS x context cross-attn\n(4 CLS x 8 dim)")
    box(10, 30, 30, 6, "CLS x tiles cross-attn")
    box(10, 22, 30, 6, "latent FFN -> 32-dim actor latent")

    # Action heads + temperature + minGRU.
    box(2, 12, 12, 8, "5 action heads\n(bt, bd, cd,\nbrt, md)", fc="#ffe9d6")
    box(16, 12, 12, 8,
        "head_temperature\n5 learned temps\n(state dependent)", fc="#ffe9d6")
    box(30, 12, 16, 8,
        "minGRU recurrent\nstate (H=24)\nh_gate, h_cand -> h", fc="#ffe9d6")

    arrow(25, 58, 25, 54.2)
    arrow(25, 48, 25, 44.2)
    arrow(25, 38, 25, 36.2)
    arrow(25, 30, 25, 28.2)
    arrow(25, 22, 25, 20.2)

    # ---- CRITIC (right column) ----
    ax.text(82, 70, "Per-unit segmenting critic — model.py:1448 (CentralizedCritic)",
            ha="center", fontsize=9.5, fontweight="bold")
    box(56, 58, 50, 8,
        "Whole-map tile features (RxHxWxC=23)\n+ scalar context + per-unit context",
        fc="#fafafa")
    box(56, 48, 22, 6, "CriticTileEmbed\n(MLP -> 8d, NHWC)")
    box(80, 48, 26, 6, "Patchify 2x2 -> 32d patches\n2D RoPE")

    # Three stacked transformer blocks — visual depth showing the depth of the
    # repeated [windowed SA x 2 + global SA] block.
    stacked_box(
        56, 38, 50, 6,
        "3 x [windowed SA (5x5) x 2 + global SA]\nperi-LN + GQA(4h, 2 KV) + QK-norm",
        n_layers=3, offset=0.6,
    )

    box(56, 28, 24, 6, "8 learned global CLS\nidentity-rotated")
    box(82, 28, 24, 6, "Per-unit CLS\nactor latent + 5 action embeds\n+ unit pos")

    box(56, 20, 50, 6,
        "Cross-visibility SDPA\n(patches + scalar + CLS + units, all attend)")

    # Separate value heads: global pooled value vs per-unit values.
    box(56, 10, 24, 8,
        "Global value head\nsoftmax-pool 8 CLS\n-> V_global(s)",
        fc="#ffe9d6")
    box(82, 10, 24, 8,
        "Per-unit value head\nHL-Gauss 255-bucket soft CE\n-> V_unit(s, a)",
        fc="#ffe9d6")

    arrow(82, 58, 67, 54.2)
    arrow(82, 58, 93, 54.2)
    arrow(67, 48, 81, 44.2)
    arrow(93, 48, 81, 44.2)
    arrow(81, 38, 81, 34.2)
    arrow(81, 28, 81, 26.2)
    arrow(81, 20, 68, 18.2)
    arrow(81, 20, 94, 18.2)

    # Route the actor-latent -> per-unit-CLS arrow underneath the stacked
    # transformer block so it does not cross any other boxes. The arc dips
    # down below the SDPA fan-out before rising into the per-unit CLS box.
    ax.annotate(
        "",
        xy=(94, 28), xytext=(40, 23),
        arrowprops=dict(
            arrowstyle="->", color="#a44", lw=1.1,
            connectionstyle="arc3,rad=0.45",
            shrinkA=2, shrinkB=2,
        ),
    )
    ax.text(67, 5.5,
            "actor latent + 5 sampled actions  ->  per-unit CLS  (the SAM3-style segmenting hook)",
            ha="center", fontsize=7.5, color="#a44", style="italic")

    ax.text(55, -0.5,
            "Inference path is CUDA-graphed (model.py:2581 GraphedInference); "
            "critic encode_map is torch.compile()'d (model.py:4806).\n"
            "Actor weights ship as a stdlib forward pass under bots/tensor_inference/tensor_lib/.",
            ha="center", va="bottom", fontsize=7.5, style="italic")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _shade(hex_color: str, factor: float) -> str:
    """Return a shade of hex_color multiplied by factor (e.g. 0.88 = darker)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == "__main__":
    make_timeline(OUT / "timeline.png")
    make_flagship_runs(OUT / "flagship_runs.png")
    make_pretrain_curves(OUT / "pretrain_curves.png")
    make_architecture(OUT / "architecture.png")
    print("Wrote all figures to", OUT)
