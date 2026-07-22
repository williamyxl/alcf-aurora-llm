# Loop / monitor status

**Login-node policy:** no long-lived processes (`nohup`, `while sleep`, background monitors).  
`loop_10min.sh` is **disabled** (exits immediately).

**Allowed:** one-shot `bash harvest_once.sh` (optional `--submit`), or ask the agent to `qstat` / harvest when you say continue.

## Campaign
Phase F + Phase G complete. Queue empty.

## Highlights
| Recipe | gen_tps |
|--------|--------:|
| F4h (1-tile MoE + HBM pref) | **33.78** |
| F4 (1-tile MoE + DDR NUMA) | **33.28** |
| G1 (128k ctx, 1-tile MoE) | **32.66** |
| P14_tp2 (2-tile short) | **30.01** |
| G0 (128k ctx, 2-tile) | 28.61 |

See `BEST_RECIPE.md` and `CYCLE_LOG.md`.
