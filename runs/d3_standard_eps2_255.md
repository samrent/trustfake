# WP1 condition table -- effb0_standard_eps2_255

- split: `test`, n = 15316
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- temperature 3.3378 fitted on `predictions_effb0_standard_eps2_255_calib_clean` and frozen
- AURC/AUGRC convention: tie blocks collapsed, block-size-weighted mean over operating points
- ECE column: 15 bins, equal-mass
- **TRIVIAL BASELINE on this split: 'width==height -> fake' = 0.9785 accuracy (majority class 0.5000). Read every model accuracy against that number.**

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
                 clean  15316  0.8435  0.8570     0.9434      0.7945     52.52      39.38       39.57  0.0140    9.99%    4.27%   15270
           clean (T=1)  15316  0.8435  0.8570     0.9434      0.7945     52.52      39.38       39.57  0.1020    9.99%    4.27%   14703
              jpeg_q50  15316  0.7198  0.7751     0.8957      0.7188    140.41      95.96       96.87  0.1056   23.34%   14.95%   15269
        jpeg_q50 (T=1)  15316  0.7198  0.7751     0.8957      0.7188    140.41      95.96       96.87  0.2170   23.34%   14.95%   15098
         downscale_0.5  15316  0.7987  0.8050     0.8940      0.7793     75.66      55.75       53.88  0.0142   14.22%    6.57%   15267
   downscale_0.5 (T=1)  15316  0.7987  0.8050     0.8940      0.7793     75.66      55.75       53.88  0.1243   14.22%    6.57%   14879
pgd_linf_eps0.00784_s10  15316  0.0000  0.0000     0.0000         nan   1000.00     525.59        0.00  1.0000  100.00%  100.00%   10102
pgd_linf_eps0.00784_s10 (T=1)  15316  0.0000  0.0000     0.0000         nan   1000.00     990.97        0.00  1.0000  100.00%  100.00%     101
    ace_uint8_eps0.005  15316  0.8435  0.8570     0.7511      0.1093    329.51     129.83      316.57  0.2743   19.56%   30.78%   15282
ace_uint8_eps0.005 (T=1)  15316  0.8435  0.8570     0.7510      0.1091    333.22     130.07      320.28  0.1983   19.56%   30.78%   14396
overconf_eps0.0157_s20  15316  0.8435  0.8570     0.8456      0.4135    194.24      89.68      181.30  0.1565   16.88%   19.25%   12674
overconf_eps0.0157_s20 (T=1)  15316  0.8435  0.8570     0.8459      0.5000    156.50     156.50      143.56  0.1565   15.65%   15.65%       1
```

clean, broken out by fake class:

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
     real vs synthetic  11487  0.8308  0.7966     0.9877      0.7860     59.45      44.41       44.24  0.0158   11.26%    5.24%   11455
      real vs tampered  11487  0.7933  0.7399     0.8991      0.7230    103.65      66.78       80.64  0.0192   15.51%    9.93%   11473
```
