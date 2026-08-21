# WP4 selective moderation -- effb0_at_pgd_eps2_255

- policy fitted on CLEAN CALIB and frozen: `t_low=0.1886`, `t_high=0.8664`, temperature `0.9035`
- objective: minimize review rate subject to residual risk <= 5% (an SLA is a choice, not a fact)
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- residual risk = error rate among AUTO-DECIDED items; missed_fake = fakes auto-allowed

```
                 condition     acc  coverage  review%  resid_risk  missed_fake  false_flag
------------------------------------------------------------------------------------------
                     clean  0.8177     49.9%    50.1%       4.72%        3.88%       0.84%
                  jpeg_q50  0.8169     49.4%    50.6%       4.74%        3.88%       0.81%
   pgd_linf_eps0.00784_s10  0.7190     41.5%    58.5%      13.01%        8.28%       2.53%
        ace_uint8_eps0.005  0.8177     45.5%    54.5%       8.05%        5.80%       1.53%
    overconf_eps0.0157_s20  0.8177     80.8%    19.2%      12.63%       14.86%       5.56%
```
