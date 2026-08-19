# WP4 selective moderation -- tf_efficientnet_b0

- policy fitted on CLEAN CALIB and frozen: `t_low=0.0929`, `t_high=0.8881`, temperature `1.2785`
- objective: minimize review rate subject to residual risk <= 5% (an SLA is a choice, not a fact)
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- residual risk = error rate among AUTO-DECIDED items; missed_fake = fakes auto-allowed

```
                 condition     acc  coverage  review%  resid_risk  missed_fake  false_flag
------------------------------------------------------------------------------------------
                     clean  0.8311     43.2%    56.8%       4.65%        2.34%       1.68%
              ace_eps0.005  0.8311     17.1%    82.9%      98.13%       17.16%      16.34%
       ace_uint8_eps0.0005  0.8311     43.2%    56.8%       4.65%        2.34%       1.68%
        ace_uint8_eps0.002  0.8311     50.4%    49.6%      33.06%       17.01%      16.28%
        ace_uint8_eps0.005  0.8311     50.4%    49.6%      33.06%       17.01%      16.28%
       ace_uint8_eps0.0157  0.8311     49.6%    50.4%      33.66%       17.07%      16.31%
```
