# Final MC mean–CVaR80 scores

Reference: per-case minimum evaluation cost across the formal evaluation pool, fixed before score aggregation. Gap = 100*(cost/reference-1). Context fitness = 0.6*mean gap + 0.4*CVaR80 gap (fractional boundary mass). Overall applies the same operator to context fitness, with equal context weights. Missing/TLE measurements are excluded per policy, not imputed; see counts. Empirical reference, not an oracle. SQ, QR and GBS are published rules transferred from related models; not native fixed-patience methods.

| Policy | Cases | Overall fitness | Mean context fitness | CVaR80 context fitness |
|---|---:|---:|---:|---:|
| CBATS | 3600 | 1.051716 | 0.744387 | 1.512709 |
| BATS | 3600 | 1.679166 | 1.254576 | 2.316050 |
| capped_base_stock | 3600 | 2.156711 | 1.520765 | 3.110630 |
| codex-astra-urgency-correction | 3600 | 3.339479 | 2.215486 | 5.025470 |
| gbs | 3600 | 4.891192 | 3.407355 | 7.116948 |
| myopic | 3600 | 5.654643 | 4.319557 | 7.657272 |
| qr | 3600 | 13.031594 | 10.344767 | 17.061835 |
| base_stock | 3600 | 50.843250 | 37.399084 | 71.009498 |
| sq | 3600 | 53.340534 | 38.667200 | 75.350536 |
| constant_order | 3600 | 165.908539 | 129.550021 | 220.446316 |
