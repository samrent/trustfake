# WP4 selective moderation -- vit_l14_e2e

- policy fitted on CLEAN CALIB and frozen: `t_low=0.3297`, `t_high=0.5918`, temperature `0.9107`
- objective: minimize review rate subject to residual risk <= 5% (an SLA is a choice, not a fact)
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- residual risk = error rate among AUTO-DECIDED items; missed_fake = fakes auto-allowed

```
                 condition     acc  coverage  review%  resid_risk  missed_fake  false_flag
------------------------------------------------------------------------------------------
                     clean  0.9323     95.2%     4.8%       5.46%        5.00%       5.40%
        ace_uint8_eps0.005  0.9323     90.8%     9.2%       7.46%        6.85%       6.70%
              ace_eps0.005  0.9323     63.9%    36.1%      10.60%        6.85%       6.70%
   pgd_linf_eps0.00784_s10  0.0000    100.0%     0.0%     100.00%      100.00%     100.00%
    overconf_eps0.0157_s20  0.9323    100.0%     0.0%       6.78%        6.85%       6.70%
```
