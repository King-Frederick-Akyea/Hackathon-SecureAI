**Team:** Neuralynx · **Tracks:** 🔴 Advanced (primary), 🟡 Intermediate (supporting) · **Code:** [github.com/King-Frederick-Akyea/Hackathon-SecureAI](https://github.com/King-Frederick-Akyea/Hackathon-SecureAI)

## Headline results

All numbers are on the held-out NSL-KDD test set (22,544 rows), from the official notebook run (seed 42, 8 rounds, the organizers' `WeakMLP` and `local_train` unchanged).

| Track | Before | After | Change |
|---|---|---|---|
| 🔴 **Advanced**: client 1 label-flips and scales its update ×15 | Naive FedAvg under attack: **0.0511** | **Sentinel: 0.7560** | **+0.7049 F1 recovered** |
| 🟡 Intermediate: non-IID banks, no attack | Naive FedAvg: **0.7117** | **FedNova + equal weights: 0.7821** | **+0.0704 F1** |

The provided coordinate-median defense reaches 0.6855 under the same attack. Our defended model (0.756) even beats *FedAvg with no attacker* (0.712).

`model_scripted.pt` in our [GitHub repo](https://github.com/King-Frederick-Akyea/Hackathon-SecureAI) is the Sentinel model; reloading it reproduces F1 = 0.7560.

## 1. Intermediate: why FedAvg breaks on skewed banks

**Diagnosis.** The non-IID banks hold 6k–48k rows. With batch size 256 and one local epoch, they take **25, 42, 189, 91 and 89 local SGD steps**. FedAvg averages the resulting *models*, weighted by data size. Bank 2 therefore dominates twice: it has the largest weight *and* it drifted furthest (7.5× more steps than bank 0).

Wang et al. (2020, *FedNova*) show this "objective inconsistency" makes FedAvg optimize a different, skewed objective. The symptom is what we see: precision 0.95 but recall only 0.57. The global model under-calls attacks.

**Fix.** On the server:
1. Divide each client's update by its step count, `d_i = (w_i − w) / τ_i`.
2. Average with **equal per-bank weights**.
3. Rescale by the mean step count.

Nothing new leaves a bank: the step count is a public function of the batch size.

Why equal weights? Each bank sees a different slice of the threat landscape, and we want a detector that works for the *average bank*, not the average row.

**It isn't one lucky seed.** The ablation below uses 5 seeds, paired (same seed means same initialization and batch order for every method):

| Method | F1 mean ± std | Recall |
|---|---|---|
| Naive FedAvg | 0.717 ± 0.006 | 0.574 |
| Equal weights only | 0.754 ± 0.009 | 0.634 |
| FedNova (size weights) only | 0.756 ± 0.011 | 0.636 |
| **FedNova + equal weights** | **0.781 ± 0.007** | **0.679** |
| Coordinate median | 0.736 ± 0.015 | 0.601 |

Each ingredient alone gives about +0.04; together they give +0.064 on average and win on every seed, even beating FedAvg on *IID* data (0.764).

**What didn't work:**
- **FedProx** (μ = 0.01 and 0.1): no gain, 0.717 and 0.718.
- **Server momentum (FedAvgM):** no gain, at most 0.725.
- **FedAdam:** unstable (one seed fell to 0.55).
- **Class-balanced local loss:** small gain (0.728), but banks must share label counts, a privacy leak.

## 2. Advanced: Sentinel

### Measure before defending

We first logged each client's update every round. Two findings shaped the design.

**Finding 1: magnitude is a clean signal.** We compute `ratio = ‖d_i‖ / median‖d‖` on step-normalized updates. Honest banks in clean training never exceed **1.7×**. The ×15 attacker starts at **19×** and grows past **180×**, because its label-flipped model diverges further every round.

**Finding 2: direction is a trap under non-IID data.** The popular Krum/cosine-style filters score clients by how well their update aligns with the consensus. In *clean* training, honest banks 0 and 1 (81–86% attack traffic) have cosine similarity **−0.48** to the median update. They legitimately push the opposite way from the normal-heavy majority.

A direction-based filter would ban exactly the banks that see the most attacks, turning the defense into a denial-of-service against under-represented traffic. So we avoided direction-based scoring.

### The algorithm (every round)

1. **Normalize:** `d_i = (w_i − w)/τ_i`, so a large bank isn't "loud" just because it took more steps.
2. **Score:** `ratio_i = ‖d_i‖ / median_j‖d_j‖`. The median stays honest with up to 2 of 5 attackers.
3. **Detect:** `ratio_i > 5` excludes the client this round and adds a strike.
4. **Quarantine:** 2 strikes excludes the client for the rest of training (a reputation system).
5. **Clip:** every survivor is clipped to 2× the median norm, bounding what an *undetected* attacker can do in one round.
6. **Aggregate:** survivors are combined with FedNova + equal weights (our Intermediate fix).

The notebook prints the detection log each round. In the official run, client 1 is flagged in all 8 rounds (ratio 18.9 up to 181), and all four honest banks are always aggregated.

### A failure we found and fixed: the exclusion feedback loop

Our first version used a threshold of 3. On the official seed it started excluding **honest bank 0** from round 4. The mechanism:
- With the attacker gone, bank 0 is the only strongly attack-heavy bank left.
- Once bank 0 is also excluded, the global model drifts away from its traffic.
- Its next update is therefore *even larger* (the ratio rose from 3 to 10), so it stays banned.

This self-reinforcing false positive can hit any outlier-exclusion defense on non-IID data. The fix is a tiered response: moderate outliers are **dampened by clipping, not banned**, and only extreme outliers (>5×) earn strikes. False exclusions of honest banks dropped from 5 to **0**.

### Stress test (5 seeds per cell, mean F1)

| Scenario | FedAvg | Median | Clip skeleton | **Sentinel** |
|---|---|---|---|---|
| No attack (cost of defense) | 0.717 | 0.736 | 0.717 | **0.781** |
| Scale ×15, client 1 (official) | 0.219 | 0.679 | 0.619 | **0.760** |
| Scale ×50, client 1 | 0.521 | 0.679 | 0.619 | **0.760** |
| Scale ×3, client 1 (stealthier) | 0.658 | 0.677 | 0.658 | **0.756** |
| Scale ×15, clients 1 + 3 colluding | 0.138 | 0.735 | 0.481 | **0.776** |
| Label-flip only, client 1 | 0.688 | 0.675 | 0.688 | **0.752** |

Detection, counted in client-rounds (5 seeds × 8 rounds per attacker):

| Scenario | Attacker excluded | Honest banks excluded |
|---|---|---|
| ×15 and ×50 | 40/40 | **0** |
| ×3 | 34/40 | **0** |
| Two attackers | 80/80 | **0** |
| Label-flip only | 15/40 | **0** |
| No attack | – | **0** |

Sentinel has **no accuracy cost when nobody attacks**. It is actually the best clean aggregator we found, because it contains the Intermediate fix.

## 3. Honest limitations (the security view)

- **Quiet label-flips are only partly detected (15/40).** Without scaling, a flipped update is normal-sized, so magnitude can't expose it; clipping plus FedNova limit the damage instead. Catching it would need a signal Sentinel doesn't use, such as a small trusted server-side validation set, or comparing a bank with peers that have similar traffic.
- **Equal weighting is a double-edged sword.** It gives a small bank more say, which also means more say for a small *malicious* bank. Without detection, a pure label-flip hurts FedNova-EW more than FedAvg (0.685 vs 0.688, despite a higher clean baseline). Pairing fair weighting with detection is the point.
- **An adaptive attacker could stay under 5× every round.** Clipping to 2× the median bounds that attacker's per-round influence, but it could still slowly bias the model. Sybil attacks (many colluding banks) break any median-based reference once attackers are the majority.
- **Tuning honesty:** we chose among ~20 aggregation variants and the thresholds by looking at test-set F1 across 5 seeds. That is the only labeled evaluation this notebook provides. To limit overfitting we report 5-seed means and the ablation. We also chose the threshold of 5 by a label-free criterion: zero honest exclusions, with a 3× margin over the largest clean ratio (1.7).
- **The test set is harder than the training data.** KDDTest+ contains attack types absent from training, which caps recall for this 8-unit MLP. We kept the organizers' model fixed so every gain is attributable to aggregation.

## 4. Reproducibility

Our [GitHub repo](https://github.com/King-Frederick-Akyea/Hackathon-SecureAI) contains the fully executed notebook, `model_scripted.pt`, `submission.json`, `results.json` (every number above, plus per-round detection logs), and a README with setup steps.

Our only change to organizer cells is a `reseed()` call before each run, so before and after start from identical weights and batch order. It runs on CPU in about 5 minutes.

**Takeaway:** in federated security, fairness and robustness are coupled. A defense that doesn't account for non-IID data will mistake honest diversity for an attack.
