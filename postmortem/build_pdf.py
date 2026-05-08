"""Assemble the postmortem PDF from prose + figures."""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak,
    KeepTogether, Table, TableStyle,
)

OUT = Path("/home/marvin/Documents/repositories/cb-bc/postmortem")


styles = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=styles["Heading1"],
                   fontName="Helvetica-Bold", fontSize=16, leading=20,
                   spaceBefore=8, spaceAfter=6, textColor=HexColor("#1a2a4a"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"],
                   fontName="Helvetica-Bold", fontSize=11.5, leading=14,
                   spaceBefore=8, spaceAfter=3, textColor=HexColor("#243b6e"))
BODY = ParagraphStyle("Body", parent=styles["BodyText"],
                      fontName="Helvetica", fontSize=9.5, leading=13,
                      alignment=TA_JUSTIFY, spaceAfter=4,
                      firstLineIndent=0)
CAP = ParagraphStyle("Caption", parent=styles["BodyText"],
                     fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
                     alignment=TA_CENTER, textColor=HexColor("#444"),
                     spaceBefore=2, spaceAfter=8)
META = ParagraphStyle("Meta", parent=styles["BodyText"],
                      fontName="Helvetica", fontSize=8.5, leading=11,
                      alignment=TA_LEFT, textColor=HexColor("#555"),
                      spaceAfter=6)
CODE = ParagraphStyle("Code", parent=styles["BodyText"],
                      fontName="Courier", fontSize=8.2, leading=11,
                      textColor=HexColor("#333"))


def fig(path, width_in, caption=None):
    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    target_w = width_in * inch
    img.drawWidth = target_w
    img.drawHeight = target_w * (ih / iw)
    items = [img]
    if caption:
        items.append(Paragraph(caption, CAP))
    return KeepTogether(items)


def fig_row(paths, total_width_in, caption=None, gap_in=0.08):
    """Render multiple images horizontally adjacent in a single row.

    Each image is sized to its column width and aspect-ratio preserved; the
    row is rendered with a Table so reportlab balances the cells.
    """
    n = len(paths)
    cell_w_in = (total_width_in - gap_in * (n - 1)) / n
    cells = []
    for path in paths:
        img = Image(str(path))
        iw, ih = img.imageWidth, img.imageHeight
        target_w = cell_w_in * inch
        img.drawWidth = target_w
        img.drawHeight = target_w * (ih / iw)
        cells.append(img)
    col_widths = []
    for i in range(n):
        col_widths.append(cell_w_in * inch)
        if i < n - 1:
            col_widths.append(gap_in * inch)
    row = []
    for i, c in enumerate(cells):
        row.append(c)
        if i < n - 1:
            row.append("")
    table = Table([row], colWidths=col_widths, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    items = [table]
    if caption:
        items.append(Paragraph(caption, CAP))
    return KeepTogether(items)


def p(s):
    return Paragraph(s, BODY)


def h1(s):
    return Paragraph(s, H1)


def h2(s):
    return Paragraph(s, H2)


def build():
    doc = SimpleDocTemplate(
        str(OUT / "tensor_bot_postmortem.pdf"),
        pagesize=LETTER,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="Cambridge Battlecode tensor bot — postmortem",
        author="Carson Burke",
    )
    story = []

    story.append(h1("Cambridge Battlecode tensor bot &mdash; postmortem"))
    story.append(Paragraph(
        "An RL agent for Battlecode 2026, trained March&ndash;April 2026. "
        "Stdlib-only inference, transformer actor and per-unit segmenting critic, "
        "behavior-cloned from human replays, fine-tuned with PPO/PMPO under a "
        "vectorized self-play league.",
        META,
    ))

    story.append(h2("Preface"))
    story.append(p(
        "This is an AI-generated article but has been prompted and "
        "reviewed by me (Carson) for accurate data and details and "
        "expression of my own views and approaches."
    ))

    # ABSTRACT
    story.append(h2("Abstract"))
    story.append(p(
        "We trained a neural Battlecode agent end-to-end under a deployment "
        "constraint that the inference path use only the Python standard "
        "library. Over five weeks the project went from a small MLP self-play "
        "bot to a transformer actor with an attention-based per-unit value "
        "head, trained with PMPO on top of a centralized critic that uses "
        "HL-Gauss distributional value targets "
        "(Farebrother et&nbsp;al.,&nbsp;2024) in symlog space. Behavior "
        "cloning on the top-rated human replay corpus gave a usable cold "
        "start; encoder pretraining and an income-shaped HL-Gauss "
        "critic-pretrain stage unblocked PPO from learning the world model "
        "and the value function simultaneously. Direct influences: AlphaStar "
        "(multi-head categorical actor, league self-play, BC bootstrap), "
        "MAPPO/SMAC (centralized critic with decentralized actor), and SAM3 "
        "(per-token CLS queries that turn one shared backbone into per-agent "
        "outputs). This document is an honest account of what worked, what "
        "we abandoned, and the engineering details that mattered."
    ))

    # FIGURE 1 — timeline
    story.append(fig(
        OUT / "timeline.png", 7.0,
        caption="Figure 1. Project timeline. Bands are anchored to the "
                "actual git-commit dates in <i>cb-bc</i>; lanes show how "
                "each subsystem evolved in parallel.",
    ))

    story.append(h2("1.&nbsp;&nbsp;Problem and constraints"))
    story.append(p(
        "Battlecode is a partially observable RTS where each team controls a "
        "fleet of heterogeneous units (harvesters, conveyors, gunners, "
        "sentinels, bridges, and so on) on a tile grid. Two project-wide "
        "constraints shaped every decision: <b>(a)</b> inference code under "
        "<font face='Courier'>bots/tensor_inference/</font> may not import "
        "torch or any third-party package, and <b>(b)</b> the runtime budget "
        "is roughly 2&nbsp;ms of CPU per unit per round. Training is "
        "unconstrained and uses PyTorch, but the trained weights must "
        "re-implement as a stdlib forward pass with bit-for-bit parity."
    ))
    story.append(p(
        "The action space is multi-head categorical with five heads sampled "
        "in a causal hierarchy "
        "(<font face='Courier'>model.py:2499</font>): "
        "<i>build_dir</i> (9), <i>build_type</i> (13, including a "
        "<i>none</i> no-op), <i>conv_dir</i> (4 cardinal facings, legal only "
        "for conveyors/splitters/armoured), <i>bridge_target</i> (28 offsets "
        "in a radius-3 ring), and <i>move_dir</i> (9). Per-head temperatures "
        "are predicted by a learned linear head and squashed into "
        "[1.0,&nbsp;3.0] via a log-space sigmoid "
        "(<font face='Courier'>model.py:94&ndash;115, 3134&ndash;3140</font>). "
        "Action masking runs from the observation alone "
        "(<font face='Courier'>main.py:803</font> in inference, "
        "<font face='Courier'>shared.py:432</font> and "
        "<font face='Courier'>model.py:415</font> in training) so masks "
        "agree across the deploy and training paths."
    ))

    # SECTION 2 — stdlib tensor_lib
    story.append(h2("2.&nbsp;&nbsp;Stdlib-only inference: <i>tensor_lib</i>"))
    story.append(p(
        "<font face='Courier'>bots/tensor_inference/tensor_lib/</font> is a "
        "from-scratch forward-pass library in pure Python. The "
        "<font face='Courier'>Tensor</font> class "
        "(<font face='Courier'>tensor_lib/tensor.py:16&ndash;299</font>) "
        "wraps a flat <font face='Courier'>list[float]</font> and a shape "
        "tuple, leaning on C-backed primitives wherever the standard library "
        "exposes them: <font face='Courier'>math.sumprod</font> (aliased "
        "<font face='Courier'>_DOT</font>), <font face='Courier'>"
        "operator.mul</font> with <font face='Courier'>map</font>, and "
        "stride slicing for matmul column extraction. Linear, RMSNorm, "
        "LayerNorm, Sigmoid/Tanh/SiLU/ReLU, Softmax, Sequential, and a "
        "fused Linear+RMSNorm+activation kernel all live alongside it "
        "(<font face='Courier'>tensor_lib/nn.py:504</font>). The deployed "
        "actor is <font face='Courier'>ActorModel</font> "
        "(<font face='Courier'>tensor_lib/actor.py:260</font>): tile "
        "embedding, a tile self-attention block with SuSIE additive "
        "positional bias, a learned look-away register, exclusive "
        "self-attention (XSA), per-tile FFN with LeakyReLU&sup2;, and "
        "cross-attention onto four CLS tokens. <font face='Courier'>"
        "_build_infer_cache</font> "
        "(<font face='Courier'>actor.py:427</font>) bakes the same tied "
        "weights the torch model produces; "
        "<font face='Courier'>_get_shared_model</font> "
        "(<font face='Courier'>main.py:210</font>) caches one model across "
        "all units keyed by file mtime. The bot disables cyclic GC "
        "(<font face='Courier'>main.py:11</font>) and reuses preallocated "
        "<i>memoryview</i> slabs for QKV, attention, and softmax buffers."
    ))
    story.append(p(
        "This is the project's most distinctive engineering decision. It "
        "narrowed the model space&mdash;no sparse ops, no attention kernels "
        "we could not reimplement&mdash;but let us deploy straight from a "
        "JSON weights file inside the 2&nbsp;ms-per-unit budget."
    ))

    # SECTION 3 — architecture
    story.append(fig(
        OUT / "architecture.png", 7.0,
        caption="Figure 2. The deployable actor (left) feeds its 32-dim "
                "latent and the five sampled actions into the per-unit CLS "
                "tokens of the centralized critic (right). The critic is "
                "training-only; only actor weights ship.",
    ))

    story.append(h2("3.&nbsp;&nbsp;Actor and segmenting critic"))
    story.append(p(
        "The actor "
        "(<font face='Courier'>BuilderMLP</font>, <font face='Courier'>"
        "model.py:2842</font>) is a small CLS-attention transformer over "
        "25 tile tokens (5&times;5 visibility) plus a 20-dim global vector. "
        "Tile self-attention "
        "(<font face='Courier'>_tile_self_attention</font>, "
        "<font face='Courier'>model.py:2975</font>) scores tokens with an "
        "RBF kernel &minus;&gamma;&Vert;q&minus;k&Vert;&sup2; with learnable "
        "&gamma;, plus a SuSIE additive sinusoidal position bias "
        "(<font face='Courier'>model.py:364</font>) and a zero-initialized "
        "look-away register so the softmax has a calibrated distance sink. "
        "The RBF + SuSIE formulation comes from Artemis Cofer's "
        "<font face='Courier'>rbf_attention</font> reference "
        "(github.com/4rtemi5/rbf_attention); on the stdlib inference path "
        "it ran roughly <b>20% cheaper</b> than dot-product softmax "
        "attention with no measurable policy-quality regression in "
        "back-to-back ablations, which made it the right call for a budget "
        "measured in milliseconds. After SDPA, exclusive self-attention "
        "(XSA, <font face='Courier'>model.py:686</font>) strips the "
        "self-V-aligned component of each tile's output, so each tile "
        "absorbs only information orthogonal to what it already carries. "
        "Two CLS cross-attention stages collapse the grid into a 32-dim "
        "actor latent. A 24-dim minGRU-style "
        "<i>(1&minus;z)&middot;h+z&middot;tanh(...)</i> recurrent state "
        "carries context across rounds "
        "(<font face='Courier'>model.py:3161</font>)."
    ))
    story.append(p(
        "The critic "
        "(<font face='Courier'>CentralizedCritic</font>, "
        "<font face='Courier'>model.py:1448</font>) is a swin-style "
        "transformer over the entire map: tiles are MLP-embedded, packed "
        "into 2&times;2 patches, and processed by three blocks of "
        "<i>windowed-SA &times; 2 + global-SA</i> with grouped-query "
        "attention (4 heads, 2 KV groups), peri-LN normalization, and "
        "partial 2D RoPE. The SAM3 inspiration is the per-unit CLS builder "
        "(<font face='Courier'>CriticUnitCLSBuilder</font>, "
        "<font face='Courier'>model.py:1323</font>): each unit gets its own "
        "CLS token from <i>actor_latent</i> &oplus; per-head action "
        "embeddings (so the critic conditions on the sampled actions), and "
        "every per-unit CLS, eight learned global CLS tokens, the scalar "
        "context token, and the patch tokens go through one "
        "fully-cross-visible cuDNN SDPA call "
        "(<font face='Courier'>CriticGlobalSA</font>, "
        "<font face='Courier'>model.py:1084</font>). The action-slice of "
        "the CLS content projection is zero-initialized so the critic "
        "starts identical to a no-action baseline; the action channel only "
        "opens as gradient flows. One shared backbone thus produces a "
        "separate per-agent value, the same trick SAM3 uses to turn a "
        "shared image encoder into per-token outputs by giving each token "
        "its own query."
    ))
    story.append(p(
        "The value head is a 255-bucket HL-Gauss loss (Farebrother "
        "et&nbsp;al.,&nbsp;2024) with buckets uniformly spaced in symlog "
        "space over [&minus;9,&nbsp;9] "
        "(<font face='Courier'>model.py:153&ndash;167</font>). Targets are "
        "erf-CDF differences "
        "(<font face='Courier'>_ghl_target_probs</font>, "
        "<font face='Courier'>model.py:191</font>) forced to fp32 even "
        "under bf16 autocast to avoid bin-mass distortion. Versus MSE, the "
        "distributional head was a clear win on optimization stability and "
        "explained variance for both end-to-end PPO and the supervised "
        "critic-pretrain stage (Section&nbsp;5)."
    ))

    # SECTION 4 — training pipeline
    story.append(h2("4.&nbsp;&nbsp;Training pipeline and throughput"))
    story.append(p(
        "Training is a single GPU server multiplexing many concurrent "
        "games. <font face='Courier'>collect_games_batched</font> "
        "(<font face='Courier'>train_batched.py:615</font>) launches one OS "
        "subprocess per game with "
        "<font face='Courier'>tensor_proxy/main.py</font> on side A and "
        "either <font face='Courier'>tensor_proxy_opp/main.py</font> "
        "(a frozen league opponent) or "
        "<font face='Courier'>tensor_dummy/main.py</font> (a scripted "
        "no-op curriculum) on side B. Proxy bots talk to the GPU server "
        "through a single-producer/single-consumer ring buffer in "
        "<font face='Courier'>/dev/shm</font> "
        "(<font face='Courier'>shared_collector.py:24</font>) with tiny "
        "wakeup pipes&mdash;no Python pickling on the hot path. Maps and "
        "team assignments are randomized per game with 50% side-swap "
        "probability "
        "(<font face='Courier'>train_batched.py:316</font>), so the bot "
        "trains as both A and B; winners are normalized to the actor's "
        "perspective afterwards."
    ))
    story.append(p(
        "Inference is fully CUDA-graphed: <font face='Courier'>"
        "GraphedInference</font> "
        "(<font face='Courier'>model.py:2581</font>) captures forward + "
        "masking + Gumbel-max sampling + log-prob gather inside a single "
        "<font face='Courier'>cuda.graph</font> region. Opponent inference "
        "uses bucketed cached graphs at batch sizes "
        "(8,&nbsp;16,&nbsp;32,&nbsp;64,&nbsp;128,&nbsp;256), keyed by "
        "(weight path, hidden, mtime). The critic <i>encode_map</i> is "
        "<font face='Courier'>torch.compile</font>'d "
        "(<font face='Courier'>model.py:4806</font>); critic minibatch "
        "transfers are double-buffered across two CUDA streams "
        "(<font face='Courier'>model.py:4099</font>) so batch <i>k</i>+1 "
        "staging copies overlap batch <i>k</i> compute. A "
        "<font face='Courier'>HiddenStateBank</font> "
        "(<font face='Courier'>hidden_state_bank.py</font>) keeps the "
        "recurrent state on GPU with row-key gather/update, avoiding "
        "H2D ping-pong."
    ))
    story.append(p(
        "All training ran on a single <b>RTX&nbsp;5090</b>. Rollout "
        "throughput on the current transformer-actor + transformer-critic "
        "setup is <b>~900 samples/second</b> end-to-end (rollout samples "
        "divided by elapsed time, averaged over the most recent flagship "
        "runs: <font face='Courier'>sp_20260409_manual36</font> at 935 "
        "samples/s, <font face='Courier'>sp_20260417_115120</font> at "
        "888 samples/s). PPO update config: <b>4 epochs</b> per episode "
        "for both actor and critic, actor minibatch <b>2048</b> with "
        "TBPTT unroll <b>K=32</b>, critic minibatch sized to give "
        "<b>8 batches/epoch</b> over the rollout, <b>16 parallel "
        "envs/episode</b> (4 opponents&nbsp;&times;&nbsp;4 games). A "
        "typical ablation was a 300-episode run, completing in roughly "
        "an hour of wall clock time."
    ))

    # SECTION 5 — pretraining
    story.append(h2("5.&nbsp;&nbsp;Pretraining: BC actor + HL-Gauss critic"))
    story.append(p(
        "Both pretrain stages live in <font face='Courier'>"
        "bots/tensor/torch/pretrain/</font>. Replays are parsed by a Rust "
        "PyO3 crate "
        "(<font face='Courier'>packages/replay_parser/</font>) that returns "
        "raw bytes consumable via "
        "<font face='Courier'>torch.frombuffer</font> with zero copies; "
        "<font face='Courier'>_decode_sequential_replay</font> "
        "(<font face='Courier'>dataset.py:189</font>) zero-pads the "
        "observation axis on older replays whose obs vector predates a "
        "global-features bump, so the encoder still ingests them. The "
        "<font face='Courier'>--min-rating</font> filter defaults to "
        "<b>1500 average team Elo</b>; the most recent BC and "
        "critic-pretrain runs raised it to <b>2150</b> "
        "(<font face='Courier'>bc_actor_20260420_135121</font>, "
        "<font face='Courier'>critic_pretrain_income_20260420_145944</font>) "
        "to concentrate on top-tier macro play."
    ))
    story.append(p(
        "Actor BC "
        "(<font face='Courier'>train_bc.py</font>) trains the same "
        "<font face='Courier'>BuilderMLP</font> with masked cross-entropy "
        "on each action head and TBPTT through "
        "<font face='Courier'>BPTT_K=32</font>. Critic pretrain "
        "(<font face='Courier'>train_critic.py</font>) freezes the BC "
        "actor, precomputes per-step actor latents "
        "(<font face='Courier'>train_critic.py:1206</font>), and fits "
        "the centralized critic to discounted returns. The reward mode "
        "defaults to <i>income</i> (per-step "
        "<font face='Courier'>resources_collected_delta &times; 2.5</font> "
        "with no terminal bonus); the <font face='Courier'>--reward-mode</font> "
        "override added in commit <font face='Courier'>8121501</font> lets "
        "us match the RL-time reward without re-derivation. Targets are "
        "HL-Gauss distributions; the loss is soft cross-entropy "
        "(<font face='Courier'>train_critic.py:801, 876</font>)."
    ))
    story.append(fig(
        OUT / "pretrain_curves.png", 7.0,
        caption="Figure 4. Pretraining curves. Left: BC actor loss "
                "(symlog y) falls from 5.45 to 2.36 over 20 epochs, "
                "broken out into all five action-head cross-entropies. "
                "<i>build_type</i> drops fastest (categorical, low "
                "entropy); <i>build_dir</i> tracks build_type because "
                "they share the masking flow; <i>conv_dir</i> and "
                "<i>bridge_target</i> stay near zero because their loss "
                "is gated by build_type and the labels are very sparse; "
                "<i>move_dir</i> dominates the total loss and floors near "
                "the entropy of the human policy. Right: critic explained "
                "variance climbs to <b>0.93</b> on income returns within "
                "1.2K steps; the step-up discontinuities mark epoch "
                "boundaries (4 epochs over the kessoku replay index).",
    ))

    # SECTION 6 — rewards & credit assignment
    story.append(h2("6.&nbsp;&nbsp;Rewards: a timeline"))
    story.append(p(
        "Reward shaping went through five distinct formulations between "
        "mid-March and early April. This section walks them in order, "
        "describing what each variant rewarded and the asymptotic "
        "resources/episode it produced (the same scalar tracked in "
        "Section&nbsp;7)."
    ))
    story.append(p(
        "<b>(a) Hand-shaped placement deltas (03-16 to 03-21).</b> "
        "<font face='Courier'>shared.py:497&ndash;611</font>. A successful "
        "harvester on a fresh ore tile pays "
        "<font face='Courier'>PLACE_HARV=5.0</font>; a successful conveyor "
        "pays <font face='Courier'>PLACE_CONV/(1+effective_dist)</font>. "
        "Failed placements fall back to a delta-style proximity reward "
        "keyed on per-ore "
        "<font face='Courier'>GameState.harv_proximity</font> / "
        "<font face='Courier'>conv_proximity</font>, paying only when a "
        "unit strictly improves on the previously-claimed best for that "
        "ore; a successful placement sets the proximity entry to "
        "&minus;1 to lock out further stipends (single payment per ore). "
        "Carried the MLP era to basic harvester placement, but the "
        "asymptote stayed near ~1 resource/episode&mdash;there was no "
        "signal for what happens <i>after</i> placement (conveyor "
        "lanes, delivery)."
    ))
    story.append(p(
        "<b>(b) Outcome-only training "
        "(<font face='Courier'>sp_outcome_20260330</font>).</b> Pure "
        "terminal win/loss reward, no shaping. The signal was too sparse "
        "for the 200-round horizon: resources/episode never left the "
        "floor, and the run sits as the lowest curve in Figure&nbsp;5."
    ))
    story.append(p(
        "<b>(c) Income shaping "
        "(<font face='Courier'>sp_income_20260330c</font>).</b> Per-step "
        "<font face='Courier'>resources_collected_delta &times; 2.5</font> "
        "with no terminal bonus, broadcast per-round. Asymptote ~17 "
        "resources/episode&mdash;a clear step over outcome-only, but "
        "income alone could not credit the infrastructure that produced "
        "the collection (conveyor topology, delivery routes)."
    ))
    story.append(p(
        "<b>(d) BFS flow distance + KL-LR "
        "(<font face='Courier'>sp_20260401</font>).</b> Per-conveyor "
        "distance-weighted credit via BFS over the placed graph, paired "
        "with the KL-adapted learning rate. Asymptote ~35 "
        "resources/episode, roughly 2&times; income-only&mdash;the bot "
        "now had a gradient pointing at the conveyor structure itself, "
        "not just the harvester delta."
    ))
    story.append(p(
        "<b>(e) Per-team flow + delivery rewards (04-01 onward, final).</b> "
        "<font face='Courier'>flow_reward.py</font>. The Rust parser "
        "exposes per-turn <font face='Courier'>transfer_events</font> and "
        "<font face='Courier'>resources_collected</font> scalars; "
        "<font face='Courier'>compute_dense_flow_rewards</font> "
        "(<font face='Courier'>flow_reward.py:83</font>) sums weighted "
        "conveyor hops and core-delivery deltas into a per-round team "
        "reward, broadcast identically to every active unit "
        "(<font face='Courier'>augment_trajectories_with_flow_rewards</font>, "
        "<font face='Courier'>flow_reward.py:116</font>). This is the "
        "formulation underneath every flagship run from 04-01 forward; "
        "it carried the RBF/peri-LN actor "
        "(<font face='Courier'>sp_20260409_manual36</font>) to the "
        "project's best asymptote at <b>~305 resources/episode</b> and "
        "is still the active reward for the cross-visibility SDPA "
        "critic run."
    ))
    story.append(p(
        "Two decisions held uniform across all five variants. <b>(i)</b> "
        "No reward normalization, only advantage normalization "
        "(<font face='Courier'>model.py:2429</font>); a running-mean "
        "reward normalizer was tried and reverted (commits "
        "<font face='Courier'>77aecce</font>&nbsp;&rarr;"
        "<font face='Courier'>123f851</font>) after interacting badly "
        "with PMPO sign-weighted updates. <b>(ii)</b> Per-unit GAE over "
        "the broadcast team reward, with the per-unit critic baseline "
        "subtracting &ldquo;what a typical unit at this state would "
        "contribute&rdquo;&mdash;the MAPPO/SMAC compromise. There is no "
        "explicit reward decomposition (a known weakness when one or two "
        "units dominate the scoring delta), but the segmenting critic "
        "(Section&nbsp;3) at least lets the baseline <i>differ</i> per "
        "unit."
    ))

    # SECTION 7 — flagship runs / results
    story.append(h2("7.&nbsp;&nbsp;Results"))
    story.append(p(
        "Our primary scalar metric is <i>mean cumulative harvested "
        "resources per episode</i>: a stable, comparable signal across "
        "opponents (winrate is not&mdash;it converges to ~50% in balanced "
        "self-play by construction) and the literal quantity the bot "
        "maximizes under the income-shaped reward. Figure&nbsp;5 overlays "
        "nine flagship transformer-era runs, one per architectural regime, "
        "spanning the project's lifetime. The trend is consistent: each "
        "step raised asymptotic resources/episode by roughly an order of "
        "magnitude. The income-shaping run "
        "(<font face='Courier'>sp_income_20260330c</font>) tops out near "
        "17; the flow-distance run "
        "(<font face='Courier'>sp_20260401</font>) jumps to ~35; the "
        "CLS-attention actor "
        "(<font face='Courier'>sp_20260405_193453</font>, 1001 episodes) "
        "ramps from ~1 to ~120; the RBF/peri-LN actor "
        "(<font face='Courier'>sp_20260409_manual36</font>) plateaus near "
        "<b>305 resources/episode</b>; the encoder-pretrained run "
        "(<font face='Courier'>sp_encoder_pt_20260407_004234</font>) "
        "reaches ~190; and the most recent cross-visibility SDPA critic "
        "(<font face='Courier'>sp_20260417_115120</font>) is still "
        "ramping. The RBF/peri-LN run is our cleanest stability "
        "datapoint: no policy collapse, KL-adapted learning rate stays "
        "in band "
        "(<font face='Courier'>train_batched.py:1703</font>, bounds "
        "[3e-5, 3e-3]), and resources/episode keeps climbing as the "
        "league hardens."
    ))
    story.append(fig(
        OUT / "flagship_runs.png", 7.0,
        caption="Figure 5. Mean cumulative harvested resources per episode "
                "across nine flagship training runs (20-episode rolling "
                "mean, symlog y-axis). Each run is from a distinct "
                "architectural regime; together they trace roughly two "
                "orders of magnitude of policy improvement from late "
                "March through April.",
    ))

    # Plateau / specialization failure discussion.
    story.append(p(
        "<b>Why the curves plateau.</b> The most informative pathology in "
        "late-run replays is shown in Figure&nbsp;6: even after long "
        "training, agents fail to <i>specialize into complementary "
        "roles</i>. They all build toward the same ore deposit&mdash;"
        "almost always in a straight line vertically or horizontally from "
        "the core&mdash;and ignore equally accessible ore in other "
        "directions. This falls out of two design choices interacting. "
        "<b>(i)</b> The actor is small (a 32-dim CLS-attention "
        "transformer with five categorical heads, parameter count "
        "deliberately constrained by the 2&nbsp;ms/unit deploy budget). "
        "Without enough capacity, near-identical local observations "
        "collapse to a near-deterministic policy: every unit that sees a "
        "similar 5&times;5 view picks the same direction. "
        "<b>(ii)</b> Credit is assigned at the team level&mdash;the "
        "per-team flow + delivery reward in "
        "<font face='Courier'>flow_reward.py</font> is broadcast "
        "identically to every active unit. The segmenting critic gives "
        "each unit a different baseline, but with one global reward "
        "signal there is no gradient pressure to <i>distinguish</i> "
        "&ldquo;you should have gone east while your sibling went "
        "west.&rdquo; Both factors push toward the easy attractor: a "
        "single straight conveyor lane to the nearest ore. Fixing this "
        "needs (a) a larger actor with diversified per-unit context "
        "(e.g., a unit-id embedding or a learned role token) and "
        "(b) per-unit reward decomposition&mdash;crediting each unit "
        "with the throughput it actually carries&mdash;rather than "
        "broadcasting the team total."
    ))
    story.append(fig_row(
        [
            OUT / "policy_specialization.png",
            OUT / "policy_late_red.png",
            OUT / "policy_late_blue.png",
        ],
        7.0,
        caption="Figure 6. Three late-run replay frames. <i>Left:</i> a "
                "single straight north-south lane from the core toward "
                "the nearest ore cluster, ignoring equally accessible ore. "
                "<i>Center:</i> the same pattern&mdash;harvesters and "
                "conveyors stack into one collinear chain. <i>Right:</i> a "
                "similar convergence on the "
                "same axis-aligned conveyor topology. Across frames the "
                "policy fails to specialize per-unit: with one team-level "
                "reward and a small actor, similarly-conditioned units "
                "collapse to the same deterministic action.",
    ))
    story.append(p(
        "On the optimization side, the sequence "
        "<i>NorMuon &rarr; PMPO &rarr; logit softcap + learned "
        "temperature</i> was decisive. NorMuon (Li et&nbsp;al.,&nbsp;"
        "2025) is Muon plus per-row second-moment "
        "normalization; our implementation "
        "(<font face='Courier'>normuon.py:85</font>) is a "
        "<b>fused</b> port&mdash;Newton&ndash;Schulz orthogonalization, "
        "second-moment scaling, and the shape-aware RMS rescale happen in "
        "one kernel-fused pass over the parameter tensor. Versus AdamW it "
        "won on three axes. The apples-to-apples harness "
        "(<font face='Courier'>scripts/compare_tile_sa_loss.py</font>) "
        "trains each tile-SA variant once with Adam and once with NorMuon "
        "under identical seeds; on that toy regression NorMuon converged "
        "to a lower MSE in fewer steps. In end-to-end PPO we also saw "
        "(i) a steadier actor gradient norm&mdash;the Newton&ndash;Schulz "
        "step orthogonalizes the update so spikes from a single "
        "high-variance row are damped&mdash;and (ii) lower peak GPU "
        "memory, since Muon stores only first-momentum plus a per-row "
        "second-moment scalar where AdamW carries both moments at full "
        "shape. PMPO (Zhao et&nbsp;al.,&nbsp;2025) "
        "replaces the PPO surrogate's advantage magnitude with a learned "
        "<b>separate positive/negative weighting</b> on the sign of the "
        "advantage (<font face='Courier'>model.py:2432</font>), letting "
        "positive- and negative-advantage updates be tuned independently. "
        "We adopted it after seeing the centralized critic occasionally "
        "emit magnitude-unstable advantages on rare units (the broadcast "
        "team reward pushes variance up when one unit happens to be near "
        "a delivery event). Logit softcap and learned per-head "
        "temperatures (Section&nbsp;3) replaced an earlier DreamerV3-style "
        "unimix and gave the entropy schedule a cleaner handle: we now "
        "adapt the entropy <i>coefficient</i> per head "
        "(<font face='Courier'>model.py:3859</font>, commit "
        "<font face='Courier'>517889c</font>) rather than the temperature."
    ))

    # SECTION 8 — future work
    story.append(h2("8.&nbsp;&nbsp;Future work"))
    story.append(p(
        "The two failure modes above&mdash;collapse to a single straight "
        "ore lane (Section&nbsp;7) and a ceiling on resources/episode "
        "under broadcast team reward (Section&nbsp;6)&mdash;both point at "
        "one lever: <b>more actor capacity and a richer signal to update "
        "it with</b>. Two concrete next steps."
    ))
    story.append(p(
        "<b>(1) A bigger, temporally-aware actor.</b> The current actor "
        "is deliberately tiny: one tile-SA block, two CLS cross-attentions, "
        "and a 32-dim latent. Stacking more tile-SA / context-SA layers "
        "(reusing the RBF + SuSIE structure that already runs cheaply on "
        "the stdlib path) would let the actor reason about more global "
        "structure&mdash;e.g., that an ore cluster east is worth more "
        "than a longer northbound lane&mdash;instead of collapsing every "
        "spawn to the same local greedy choice. The temporal axis is "
        "even more pressing: minGRU is a budget-driven compromise and "
        "the wrong primitive for long-horizon credit assignment in an "
        "RTS where round-30 decisions only pay out at round 200. An "
        "<b>autoregressive transformer over time</b>&mdash;a per-unit "
        "causal stack over the last K rounds with KV-cache reuse on "
        "inference&mdash;would replace the single 24-dim hidden with a "
        "real sequence model and condition each step's actions on the "
        "unit's actual build and movement history. The deploy cost is "
        "tractable: with KV caching the marginal per-round cost is one "
        "attention step per cached token, which fits the 2&nbsp;ms "
        "budget for modest K."
    ))
    story.append(p(
        "<b>(2) A world model for long-horizon credit assignment.</b> "
        "Today's team-broadcast reward and per-unit GAE squeeze a thin "
        "signal: some unit on the team caused a delivery and you are "
        "credited because you were alive. A learned world model trained "
        "on the same high-ELO replay corpus would turn that thin signal "
        "into a much richer one. The template is <b>Dreamer 4</b> "
        "(Hafner et&nbsp;al.,&nbsp;2025): a "
        "transformer-based latent world model with per-token policy and "
        "value heads, trained jointly on observations, actions, and "
        "rewards, with imagination rollouts that propagate credit back "
        "tens to hundreds of steps without touching the real environment. "
        "That buys us three things: (a) per-unit counterfactual values "
        "(&ldquo;what would the team's resources be at round 200 if "
        "<i>this</i> unit had built east instead of north?&rdquo;), the "
        "per-unit decomposition our segmenting critic currently only "
        "approximates with a baseline; (b) much more signal extracted "
        "from the fixed replay corpus than BC + HL-Gauss critic-pretrain "
        "alone&mdash;the world model generalizes over states BC has never "
        "visited; and (c) a principled handle on the long-horizon "
        "delivery dynamics (lanes that take 100+ rounds to mature) that "
        "flow-reward shaping currently papers over."
    ))

    # SECTION 9 — works cited
    story.append(h2("9.&nbsp;&nbsp;Works cited"))
    story.append(p(
        "Carion, N., Gustafson, L., Hu, Y.-T., Debnath, S., Hu, R., "
        "Suris, D., et&nbsp;al. (2025). "
        "<i>SAM 3: Segment Anything with Concepts.</i> "
        "arXiv:2511.16719."
    ))
    story.append(p(
        "Cofer, A. (2024). <i>rbf_attention</i> [Source code]. GitHub. "
        "https://github.com/4rtemi5/rbf_attention."
    ))
    story.append(p(
        "Farebrother, J., Orbay, J., Vuong, Q., Ta&iuml;ga, A.&nbsp;A., "
        "Chebotar, Y., Xiao, T., et&nbsp;al. (2024). "
        "<i>Stop Regressing: Training Value Functions via Classification "
        "for Scalable Deep RL.</i> arXiv:2403.03950."
    ))
    story.append(p(
        "Hafner, D., Pasukonis, J., Ba, J., &amp; Lillicrap, T. (2023). "
        "<i>Mastering Diverse Domains through World Models.</i> "
        "arXiv:2301.04104."
    ))
    story.append(p(
        "Hafner, D., Yan, W., &amp; Lillicrap, T. (2025). "
        "<i>Training Agents Inside of Scalable World Models.</i> "
        "arXiv:2509.24527."
    ))
    story.append(p(
        "Li, Z., Liu, L., Liang, C., Chen, W., &amp; Zhao, T. (2025). "
        "<i>NorMuon: Making Muon More Efficient and Scalable.</i> "
        "arXiv:2510.05491."
    ))
    story.append(p(
        "Vinyals, O., Babuschkin, I., Czarnecki, W.&nbsp;M., "
        "Mathieu, M., Dudzik, A., Chung, J., et&nbsp;al. (2019). "
        "<i>Grandmaster level in StarCraft II using multi-agent "
        "reinforcement learning.</i> Nature, 575(7782), 350&ndash;354."
    ))
    story.append(p(
        "Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., "
        "&amp; Wu, Y. (2022). "
        "<i>The Surprising Effectiveness of PPO in Cooperative, "
        "Multi-Agent Games.</i> NeurIPS. arXiv:2103.01955."
    ))
    story.append(p(
        "Zhao, C., Liu, Z., Wang, X., Lu, J., &amp; Ruan, C. (2025). "
        "<i>PMPO: Probabilistic Metric Prompt Optimization for Small "
        "and Large Language Models.</i> arXiv:2505.16307."
    ))

    doc.build(story)
    print("Wrote", OUT / "tensor_bot_postmortem.pdf")


if __name__ == "__main__":
    build()
