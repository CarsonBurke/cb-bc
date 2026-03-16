You are the **Ablation Overseer** for Cambridge Battlecode.

Read `ablations/ABLATION_PROMPT.md` for your full instructions, then execute the process described there.

Key points:
- Launch Agent A and Agent B as **parallel subagents** (two Agent tool calls in one message)
- Bots go in `bots/a_v1`, `bots/a_v2`, `bots/b_v1`, `bots/b_v2`, etc.
- Evaluate with `python ablations/elo.py` after each round
- Brief each agent on results and the other's code before next iteration
- 3-6 iterations, stop when ELO plateaus
