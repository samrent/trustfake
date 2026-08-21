# WP4 selective moderation -- effb0_trades_eps2_255

- policy fitted on CLEAN CALIB and frozen: `t_low=0.1295`, `t_high=0.7667`, temperature `0.7574`
- objective: minimize review rate subject to residual risk <= 5% (an SLA is a choice, not a fact)
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- residual risk = error rate among AUTO-DECIDED items; missed_fake = fakes auto-allowed

```
                 condition     acc  coverage  review%  resid_risk  missed_fake  false_flag
------------------------------------------------------------------------------------------
                     clean  0.7877     44.6%    55.4%       4.98%        3.40%       1.04%
                  jpeg_q50  0.7876     44.3%    55.7%       5.04%        3.45%       1.02%
   pgd_linf_eps0.00784_s10  0.7062     36.6%    63.4%      13.53%        7.30%       2.60%
        ace_uint8_eps0.005  0.7877     40.4%    59.6%       8.47%        5.20%       1.65%
    overconf_eps0.0157_s20  0.7877     74.0%    26.0%      12.89%       13.62%       5.45%
```
