# Research scope and implementation priorities

The central question is whether a recurrent dealer extracts decision-relevant adverse-selection information that is unavailable from its current observation and recent history. The environment fixes the latent regime within an episode and conditions executions on the dealer's quotes. The exact joint posterior provides the inference reference.

## Current evidence

The [independent replication](replication_memo.md) found a mean entropy-treatment gain of .905, with a training-seed interval [−.162, 1.972]. The predeclared criterion was not met. The [nonlinear control](nonlinear_memo.md) decoded all six posterior targets better on average than the linear trained-state decoder. These results leave both reliable recurrent learning and an incremental role for older memory unresolved.

The reference filter, reward accounting and recurrent sequence implementation have independent numerical checks. Experiment records retain all planned seeds, paired customer tapes, collection-time checkpoint hashes and separate seed/episode uncertainty. [Execution semantics](managed_execution.md) specify atomic completion and restart from initialization.

## Retained-trajectory diagnosis

The [decision audit](decision_quality.md) attributes 81.5% of the mean myopic score gap to quote-center choice across the five original entropy policies. Seed 14 combines the highest adverse-selection R² with poor quoting; its absolute decoding error is also the largest. This narrows the immediate problem to decision-relevant information and actor readout, without isolating a causal source.

A candidate next experiment is a small supervised action-score readout of frozen recurrent state, with matched history-only inputs and a fixed evaluation design. It should test center selection directly. This is a proposal, not a completed experiment or evidence that replacing the actor head will improve episode returns.

## Exact overlap and remaining identification limit

The [exact history-overlap audit](history_overlap.md) now extends the constructed history pair to retained policy trajectories. Of 281,600 eligible decisions, 15,724 have exact history-8 peers; 1,541 have peers with conflicting decisive reference actions. Matching includes inventory, time, previous quotes and outcomes. Overlap ranges from 0.61% to 18.27% by policy seed.

This establishes examples of decision-relevant information omitted by history-8, but leaves overall prevalence unidentified: 94.4% of eligible decisions lack exact peers. The protocol stops at exact matching. Approximate matching requires a separate specification and balance checks, rather than progressively looser matching after seeing outcomes. Counts are descriptive; the audit does not establish recurrent-state representation or causal policy use.

Additional training requires a specified estimand, minimum useful effect and precision target. The current five-pair variance estimate is itself uncertain. Increasing test episodes does not remove initialization variability.

## Implementation constraints

Preserve the simulator and frozen protocols when changing analysis code. Require numerical equivalence for performance optimizations and regression tests for failures that could alter a scientific conclusion. The [probe-audit optimization](compute_benchmark.md#probe-audit-efficiency-refinement) removes repeated decompression and row scans; its before/after summaries match exactly.

New dependencies, configuration layers and execution infrastructure require a demonstrated workload or failure. Additional architectures and shift studies are outside the current experiment. State interventions require a separate donor-matching design, same-belief and random controls, and tests for off-distribution states.
