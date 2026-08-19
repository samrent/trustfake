# WP4 selective moderation -- tf_efficientnet_b0

- policy fitted on CLEAN CALIB and frozen: `t_low=0.0929`, `t_high=0.8881`, temperature `1.2785`
- objective: minimize review rate subject to residual risk <= 5% (an SLA is a choice, not a fact)
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- residual risk = error rate among AUTO-DECIDED items; missed_fake = fakes auto-allowed

```
                 condition     acc  coverage  review%  resid_risk  missed_fake  false_flag
------------------------------------------------------------------------------------------
                     clean  0.8312     43.2%    56.8%       4.65%        2.34%       1.68%
                  jpeg_q50  0.7875     33.6%    66.4%       7.45%        3.98%       1.02%
             downscale_0.5  0.6762     58.2%    41.8%      22.46%        0.18%      25.95%
                      webp  0.7621     30.1%    69.9%       9.53%        5.54%       0.20%
             ace_eps0.0005  0.8311     20.1%    79.9%      49.04%        9.61%      10.08%
              ace_eps0.002  0.8311     17.4%    82.6%      92.33%       16.00%      16.06%
              ace_eps0.005  0.8311     17.1%    82.9%      98.13%       17.16%      16.34%
```
