# Research scope and implementation priorities

The central question is whether a recurrent dealer extracts decision-relevant adverse-selection information that is unavailable from its current observation and recent history. The environment fixes the latent regime within an episode and conditions executions on the dealer's quotes. The exact joint posterior provides the inference reference.

## Current evidence

The [independent replication](replication_memo.md) found a mean entropy-treatment gain of .905, with a training-seed interval [−.162, 1.972]. The predeclared criterion was not met. The [nonlinear control](nonlinear_memo.md) decoded all six posterior targets better on average than the linear trained-state decoder. These results leave both reliable recurrent learning and an incremental role for older memory unresolved.

The reference filter, reward accounting and recurrent sequence implementation have independent numerical checks. Experiment records retain all planned seeds, paired customer tapes, collection-time checkpoint hashes and separate seed/episode uncertainty. [Execution semantics](managed_execution.md) specify atomic completion and restart from initialization.

## Next analysis: prevalence of decision-relevant history

The [constructed history pair](model_design.md#gate-1-earlier-quote-evidence-changes-the-reference-response) establishes existence. It does not measure prevalence under the fitted policies. The next diagnostic should use retained trajectories and report:

- Matching overlap at fixed inventory, time and recent observations, with tolerances specified before inspecting matched outcomes.
- Differences in posterior adverse-selection risk and all-action myopic scores, including the best-versus-second-best score margin.
- The fraction of eligible decisions whose preferred reference action changes, reported per policy seed and with whole episodes as the sampling unit.

Inventory and previous policy actions already transmit historical information. Matches must account for both. Limited overlap should be reported as a limitation rather than addressed by progressively looser matching after seeing outcomes. This diagnostic is exploratory because the trajectories have already been inspected.

Additional training requires a specified estimand, minimum useful effect and precision target. The current five-pair variance estimate is itself uncertain. Increasing test episodes does not remove initialization variability.

## Implementation constraints

Preserve the simulator and frozen protocols when changing analysis code. Require numerical equivalence for performance optimizations and regression tests for failures that could alter a scientific conclusion. The [probe-audit optimization](compute_benchmark.md#probe-audit-efficiency-refinement) removes repeated decompression and row scans; its before/after summaries match exactly.

New dependencies, configuration layers and execution infrastructure require a demonstrated workload or failure. Additional architectures and shift studies are outside the current experiment. State interventions require a separate donor-matching design, same-belief and random controls, and tests for off-distribution states.
