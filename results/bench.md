| Policy | Seeds | Mean score | Std | Mean coins | Ghosts caught | Lives lost | Rounds to the clock | Falls |
|---|---|---|---|---|---|---|---|---|
| learned | 60 | **406** | 90 | 19.0 | 0.17 | 2.67 | 20/60 | 52 |
| planner | 60 | **360** | 98 | 18.9 | 0.03 | 2.15 | 39/60 | 55 |
| neutral | 60 | **0** | 0 | 0.0 | 0.00 | 2.88 | 5/60 | 54 |
| frozen | 60 | **0** | 0 | 0.0 | 0.00 | 0.00 | 0/60 | 60 |

Paired per-seed difference (learned - planner): mean +45 points, 95% bootstrap CI [+12, +79]; learned wins 29/60 seeds (0 ties), two-sided sign test p = 0.90. The mean difference is statistically significant (bootstrap CI excludes 0); the win rate is not significant (sign test): the learned policy wins fewer rounds than it loses, but wins by more.
