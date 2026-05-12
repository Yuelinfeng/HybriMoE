# Live Trace Motivation Figures

These figures are generated from live HybriMoE expert-cache traces and router traces.

Generated files:

- `fig1_cache_utility_disconnect.pdf/png`: cache residency versus issued prefetch utility.
- `fig2_prefetch_cache_pressure.pdf/png`: extra load/evict pressure as prefetch width increases.
- `fig3_dynamic_routing_instability.pdf/png`: same-layer consecutive decode expert-set Jaccard.
- `figure_data.csv`: numeric data used by the plots.

Recommended paper placement: Motivation / Observation / Pathology Analysis.
Do not present these short 9-token decode runs as final end-to-end performance evaluation.

Summary:

- prefetch=0: live_hit=0.503, issued_used=0.000, assignment_coverage=0.000, loads=1298, evicts=882, avg_jaccard=0.265
- prefetch=4: live_hit=0.499, issued_used=0.143, assignment_coverage=0.014, loads=1437, evicts=1021, avg_jaccard=0.260
- prefetch=8: live_hit=0.521, issued_used=0.128, assignment_coverage=0.033, loads=1629, evicts=1213, avg_jaccard=0.261
