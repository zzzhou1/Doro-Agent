---
name: phm
description: Analyze NASA C-MAPSS FD001 test engines with local LSTM and Transformer RUL tools
when-to-use: Use for FD001 engine inspection, RUL prediction, model comparison, or degradation trends
user-invocable: true
context: inline
allowed-tools: mcp__phm__list_phm_models,mcp__phm__inspect_engine,mcp__phm__predict_rul,mcp__phm__compare_rul_models,mcp__phm__get_model_metrics,mcp__phm__get_degradation_evidence
---

Analyze only the NASA C-MAPSS FD001 simulated test fleet.

For every engine-specific request:

1. Call `mcp__phm__inspect_engine` before prediction.
2. Stop and report the tool error if data quality fails or the engine is unknown.
3. For one requested model, call `mcp__phm__predict_rul` with that exact model.
4. For a comparison, call `mcp__phm__compare_rul_models`.
5. Call `mcp__phm__get_degradation_evidence` when the user asks why, for evidence,
   or for sensor trends.
6. Copy all numerical claims from tool results. Never invent or mentally calculate
   an RUL, interval, or metric.
7. State the dataset, engine ID, last cycle, model version, predicted RUL,
   empirical interval, validation RMSE, and data-quality result.
8. Describe the interval as empirical validation-residual coverage, not a formal
   safety confidence bound.
9. Describe sensor trends as associations, not physical causality.
10. State that FD001 is simulated and the output cannot authorize real-aircraft
    maintenance or release-to-service decisions.
