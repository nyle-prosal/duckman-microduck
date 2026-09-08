# Duck-Man strategy network: training record, including what failed

Strategy network: 76 inputs (5×5 local wall and token masks, ghost offsets and modes, nearest pellet,
power timer, lives, clock, coins left, scatter flag, heading) → 64 → 64 → 5 logits (E, W, S, N, stay),
wall-illegal moves masked. 9,413 parameters. Queried when the Duck-Man reaches a cell centre (or every
8 s if blocked). Fitness of a rollout = final score − 100 × lives lost, one full 240 s round.

| Run | Init | Method | Result | Lesson |
|---|---|---|---|---|
| run 1 (15 gens) | random | evolution strategies, pop 48, σ 0.1, lr 0.02, 2 random seeds/gen | fitness −170 → −15 best; the game rules were still being balanced during this run | pure ES from scratch learns, but slowly and noisily; rounds ended with 3 lives lost |
| run 2 (4 gens) | run-1 weights, +1 input | same, under the final rules | ≈ −120 | restarted when the rules changed (every rule change invalidates a policy) |
| **behaviour cloning** | n/a | 40 planner rounds on seeds 2000–2039, 8,928 decisions, cross-entropy, 300 epochs | 96% agreement; ≈ 312 mean score on held-out seeds 0–3 (planner 370) | a strong warm start; the network is *not* the planner (it plays worse, then better) |
| run 3 (10 gens) | cloned net | ES σ 0.03, lr 0.01, **no elitism** | fitness 23 → 100 (gen 3) → −33 (gen 9) | **FAILED as a process**: Adam-normalised steps random-walked a good policy off a cliff within 8 generations. First attempt with σ 0.1 was worse: the population mean fell from 303 to 196 in one generation |
| **run 4 (30 gens)** | run-3 generation 3 | ES σ 0.03, lr 0.005, **elitism**: the best policy on the full 6-seed pool is kept; the top individual is re-scored on the pool and adopted if it wins; revert to the elite after 3 stale generations | fitness 95 → **200** (gen 20, shipped); pool mean score 462 at best; a further 4 gens on 2026-09-07 did not beat it | elitism turned a noisy random walk into monotone progress |

`*_curve.csv` are the raw per-generation logs (fitness of the unperturbed policy, population statistics, wall
time); `curves.png` overlays all four. The shipped checkpoint is `checkpoints/strategy_final.npz`
(run 4, generation 20); `checkpoints/strategy_gen0.npz` is the untrained network used for the generation-0 clip.

What the trained policy does that the planner never does: it uses power pellets to **hunt ghosts** (+200
each), 0.25 catches per round vs 0.10 for the planner over 20 held-out seeds, at the cost of spending its
lives faster. Whether that trade is worth it is what the paired statistics in the top-level README measure.

## Round 2 (not part of the entry)

After the learned ghosts were trained we started re-evolving the Duck-Man against a mix of scripted and
learned ghosts (`train_es --ghosts mixed`, resumed from `strategy_final.npz`). One generation ran on a laptop
(best individual fitness 237.5) before memory ran out, and the HIM Arena machines we moved it to were capped
at 16 GB / 13.6 CPU cores, so no further generation completed before the deadline. The shipped strategy is the
round-1 elite benchmarked above; round 2 is listed under next ideas.
