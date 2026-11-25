# Final MC mean–CVaR80 scores

Reference: all retained formal policies in REFERENCE_COSTS.json. Gap = 100*(cost/reference-1). Context fitness = 0.6*mean gap + 0.4*CVaR80 gap (fractional boundary mass). Overall applies the same operator to context fitness, with equal context weights. Missing/TLE measurements are excluded per policy, not imputed; see counts. Empirical reference, not an oracle.

| Policy | Cases | Overall fitness | Mean context fitness | CVaR80 context fitness |
|---|---:|---:|---:|---:|
| ETIS | 3600 | 0.190326 | 0.163054 | 0.231235 |
| SCBS | 3600 | 0.445589 | 0.385036 | 0.536418 |
| pil | 3588 | 0.592917 | 0.486868 | 0.751991 |
| capped_base_stock | 3600 | 0.789061 | 0.657440 | 0.986492 |
| sqmax_ew | 3600 | 2.702039 | 2.269350 | 3.351072 |
| codex-astra-capped-survival | 3600 | 2.834103 | 2.294593 | 3.643368 |
| bsp_low | 3600 | 3.921828 | 2.796503 | 5.609815 |
| copsq | 3600 | 5.144709 | 4.838382 | 5.604199 |
| apil | 3600 | 5.261732 | 4.372930 | 6.594934 |
| weighted_base_stock | 3600 | 9.439312 | 6.425442 | 13.960117 |
| bsp_ws3 | 3600 | 12.018710 | 9.294102 | 16.105622 |
| constant_order | 3600 | 12.441097 | 8.323799 | 18.617044 |
| bsp_2d | 3600 | 16.957188 | 12.671790 | 23.385285 |
| bsp_oir | 3600 | 18.217753 | 13.246344 | 25.674866 |
| base_stock | 3600 | 19.612230 | 15.048846 | 26.457305 |
| ewa | 3598 | 40.668229 | 23.343868 | 66.654772 |
| bsp_low_ew | 3583 | 43.483605 | 20.633340 | 77.759001 |
