# WP4 selective moderation -- effb0_standard_eps2_255

- policy fitted on CLEAN CALIB and frozen: `t_low=0.4311`, `t_high=0.8956`, temperature `3.3378`
- objective: minimize review rate subject to residual risk <= 5% (an SLA is a choice, not a fact)
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- residual risk = error rate among AUTO-DECIDED items; missed_fake = fakes auto-allowed

```
                 condition     acc  coverage  review%  resid_risk  missed_fake  false_flag
------------------------------------------------------------------------------------------
                     clean  0.8435     72.9%    27.1%       5.47%        4.81%       3.17%
                  jpeg_q50  0.7198     63.1%    36.9%      11.17%        2.36%      11.73%
   pgd_linf_eps0.00784_s10  0.0000    100.0%     0.0%     100.00%      100.00%     100.00%
        ace_uint8_eps0.005  0.8435     80.7%    19.3%      19.18%        6.22%      24.75%
    overconf_eps0.0157_s20  0.8435    100.0%     0.0%      15.65%        6.22%      25.08%
```
