# Decision-relevant history under exact matching

**Identical history-8 inputs sometimes require different Bayesian reference quotes, but exact overlap covers only 5.6% of eligible decisions.** The retained data establish examples under fitted policies; they do not identify the overall prevalence of decision-relevant older history or show that recurrent PPO uses it.

## Design

The [fixed matching specification](history_overlap_protocol.md) uses all five original entropy-.01 policies and their existing 1,024-episode nonlinear-control cohort. This is exploratory analysis of previously inspected data, with no new training or trajectory collection.

Each match requires equality of all 72 recent-history features within a policy, including past quotes, abstention, outcomes, inventory and time. Peers come from distinct episodes. The eligible arrivals are 9–63, giving 56,320 decisions per policy and 281,600 overall. At arrival 8, the eight most recent observations still contain every prior execution; excluding that arrival avoids treating a complete execution history as truncated.

Reference scores use the existing one-step Bayesian objective, with inventory penalty .001, horizon 64 and zero fee. A preferred action is decisive only if its score exceeds the second-best score by more than 1e-10. A conflict means a decision has at least one exact-match peer with a different decisive preferred action. This is an existence indicator per decision, not the fraction of disagreeing pairs. No matching tolerances were relaxed after inspecting overlap.

## Results

| Policy seed | Matched decisions | Eligible decisions matched | Conflicting decisions | Conflicts among matched decisions | Mean peer-action score loss |
|---|---:|---:|---:|---:|---:|
| 11 | 497 | 0.88% | 61 | 12.27% | .002166 |
| 12 | 1,048 | 1.86% | 242 | 23.09% | .002092 |
| 13 | 342 | 0.61% | 53 | 15.50% | .000503 |
| 14 | 10,292 | 18.27% | 35 | 0.34% | .000025 |
| 15 | 3,545 | 6.29% | 1,150 | 32.44% | .001278 |

There are 15,724 matched decisions in 3,175 groups, with 1,541 conflicts. Every matched decision has a decisive reference choice at the specified threshold. Seed 14 supplies most overlap but few conflicts; aggregating across policies conceals this difference. Matching covers fewer than 2% of decisions for seeds 11–13.

The peer-action score loss averages `max_a S_i(a) - S_i(argmax_a S_j(a))` over other episodes in the same group, then equally over matched decisions. Self-pairs are excluded. This makes large groups contribute in proportion to their decisions rather than their quadratic number of pairs. Action counts compute the statistic without allocating pairwise matrices. The largest group has 839 decisions.

The [group table](../published-results/history-overlap-v1/groups.csv) retains score margins, action counts and within-group ranges. Maximum buy/sell adverse-selection target ranges are .1332, .1238, .0533, .1284 and .1525 for seeds 11–15. These are maxima over matched groups, not typical errors or fitted decoder metrics. The [summary](../published-results/history-overlap-v1/summary.json) and [episode aggregates](../published-results/history-overlap-v1/episodes.csv) retain all denominators and zero-match episodes.

## What this resolves

The constructed history pair in the model derivation is no longer the only evidence that older quote-conditioned information can change a reference action. Exact matches on observed policy trajectories provide additional examples: a policy receiving only that history-8 input must produce the same action distribution at both histories, whereas the full-posterior reference can prefer different actions.

This does not establish a recurrent-policy advantage. The conflicts concern the Bayesian reference, not the neural policy's use of memory; changing actions also changes subsequent information and inventory. Peer losses are visited-state myopic score differences, not recoverable returns. They do not isolate adverse selection from other posterior components such as expected asset value.

The unmatched 94.4% of decisions remain unidentified by this diagnostic. Low exact overlap is neither evidence for nor against older-memory relevance there. Counts describe this finite cohort, with no timestep-based confidence interval or population-prevalence claim. Broader approximate matching would need its own design and balance checks; loosening matching until an effect appears would invalidate this specification.

The next focused experiment remains a small action-score readout from frozen recurrent state, compared with equally specified history-only inputs. Evaluate quote choice and score loss on whole held-out episodes, rather than adding more latent-variable decoders. These overlap results motivate that comparison but do not predict that a replacement actor will improve episode returns.

## Reproduction

```powershell
python -m scripts.analyze_history_overlap
python -m pytest tests/test_history_overlap.py tests/test_decision_quality.py -q
python -m scripts.verify_publication
```

The original `results/nonlinear_v1` archives are required; reproduce them using the [nonlinear-control instructions](nonlinear_memo.md). Analysis verifies original published collection identities and artifact hashes, replays posterior updates and accounting from public history, and checks reconstructed windows against the saved decoder inputs. Exact grouping, episode isolation, tie handling, missing overlap and the action-count reduction have regression tests. The full suite passes 173 tests. The [new evidence snapshot](../published-results/history-overlap-v1/publication_manifest.json) preserves these results without modifying earlier snapshots.
