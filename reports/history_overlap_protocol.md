# Exact recent-history overlap diagnostic

This exploratory analysis uses the already inspected nonlinear-control cohort: policies 11–15, each with 1,024 episodes starting at 14,000,000. The matching rule and numerical threshold below are fixed before computing this diagnostic. This is not a preregistered independent test.

Within each policy, match all 72 entries of the eight most recent public observations by exact numerical equality. This includes previous quotes, abstention, outcomes, inventory and time at every position. Never match across policies or relax tolerances after seeing overlap. Use arrivals 9–63: at arrival 8 the eight observations still encode all eight previous executions, so there is no omitted outcome evidence. A group must contain at least two distinct episodes. Exact time matching permits at most one decision from an episode per group.

Replay posteriors and accounting from public histories and verify the original collection manifest before analysis. Check that reconstructed recent histories equal the saved decoder inputs. Use the existing ten-action Bayesian myopic score with penalty .001, horizon 64 and fee zero.

Report per policy:

- Eligible decisions, matched decisions and episodes, and matched-group sizes.
- Best-minus-second-best action-score margins. A reference choice is decisive only when its margin exceeds 1e-10; ties must not create apparent action conflicts.
- Decisions with a decisive choice and at least one matched peer with a different decisive choice. Give both the eligible-decision and matched-decision denominators.
- For each matched decision, the mean score loss from substituting another episode's reference action, with uniform peer weighting and self-pairs excluded. Ties use the reference's lowest-index rule. Average these losses over matched decisions, not over all pairs.
- Within-group ranges of the two fixed-quote adverse-selection targets and all ten action scores. These describe variation conditional on exact recent history; they are not decoder errors.

Retain group and whole-episode aggregates. Report a descriptive census of this finite cohort, without timestep-based confidence intervals or a population-prevalence claim. No match is missing support, not evidence of no older-history dependence. Sparse or zero overlap ends this diagnostic; approximate matching requires a separate design. Conflicting exact matches can show information omitted by the recent-history input, but cannot establish that recurrent state represents or causally uses that information. Peer-action losses on visited states are not recoverable episode returns.
