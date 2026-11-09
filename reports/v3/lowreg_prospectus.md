# Validation-driven low-regularization follow-up (separate four-fit budget)

This prospectus is committed on an isolated branch while the original eight-experiment/24-final-run study continues. No new AutoRec final-test JSON or metrics were inspected before this proposal. The motivation is validation only: seed17 hidden128 at reg1e-4 achieved validation RMSE0.861194, whereas reg1e-3 achieved0.911310. The original grids and raw outputs remain unchanged.

Hypothesis: the scale of parameter L2 penalties relative to an observed-only MEAN reconstruction loss warrants testing a lower weight-decay range. This does not assume the hypothesis succeeds. Compare hidden128/256 crossed with reg1e-5/1e-6, retaining lr0.001, input dropout0.2, max80 epochs, five-epoch validation RMSE checks, patience3 and min_delta1e-5. Three seeds17/29/43. No data split, target, masking, cold policy, rating scale or primary unclipped metric changes.

Use a separate experiment name and output_root. Final models still reinitialize/refit train+validation for validation-selected epochs, then test once. All candidates and failures are reported. This is FOUR ADDITIONAL parameter candidates, not the original equal-four-candidate comparison; disclose the added search budget and do not pick or suppress experiment variants based on their test scores. Prior v1/v2 benchmark exposure remains a limitation.
