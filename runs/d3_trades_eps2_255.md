# WP1 condition table -- effb0_trades_eps2_255

- split: `test`, n = 15316
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- temperature 0.7574 fitted on `predictions_effb0_trades_eps2_255_calib_clean` and frozen
- AURC/AUGRC convention: tie blocks collapsed, block-size-weighted mean over operating points
- ECE column: 15 bins, equal-mass
- **TRIVIAL BASELINE on this split: 'width==height -> fake' = 0.9785 accuracy (majority class 0.5000). Read every model accuracy against that number.**

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
                 clean  15316  0.7877  0.7535     0.8851      0.7761     82.55      60.00       58.21  0.0225   15.28%    7.10%   15271
           clean (T=1)  15316  0.7877  0.7535     0.8851      0.7761     82.55      60.00       58.21  0.0341   15.28%    7.10%   15271
              jpeg_q50  15316  0.7876  0.7529     0.8844      0.7730     83.41      60.53       59.06  0.0220   15.41%    7.27%   15266
        jpeg_q50 (T=1)  15316  0.7876  0.7529     0.8844      0.7730     83.41      60.53       59.06  0.0344   15.41%    7.27%   15266
         downscale_0.5  15316  0.7649  0.7156     0.8704      0.7586    100.87      71.05       70.76  0.0378   17.73%    9.40%   15270
   downscale_0.5 (T=1)  15316  0.7649  0.7156     0.8704      0.7586    100.87      71.05       70.76  0.0326   17.73%    9.40%   15270
pgd_linf_eps0.00784_s10  15316  0.7062  0.6564     0.7684      0.6795    170.51     109.68      122.36  0.0705   25.26%   18.86%   15315
pgd_linf_eps0.00784_s10 (T=1)  15316  0.7062  0.6564     0.7684      0.6795    170.51     109.68      122.36  0.0313   25.26%   18.86%   15315
    ace_uint8_eps0.005  15316  0.7877  0.7535     0.8361      0.6541    117.95      80.41       93.61  0.0722   20.05%   12.05%   15270
ace_uint8_eps0.005 (T=1)  15316  0.7877  0.7535     0.8361      0.6541    117.95      80.41       93.61  0.0511   20.05%   12.05%   15270
overconf_eps0.0157_s20  15316  0.7877  0.7535     0.8822      0.7789     81.31      59.52       56.97  0.1135   15.34%    6.92%   15266
overconf_eps0.0157_s20 (T=1)  15316  0.7877  0.7535     0.8822      0.7789     81.31      59.52       56.97  0.0751   15.34%    6.92%   15266
```

clean, broken out by fake class:

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
     real vs synthetic  11487  0.9297  0.8987     0.9787      0.8671     14.12      11.17       11.58  0.0951    2.39%    0.84%   11455
      real vs tampered  11487  0.7381  0.4794     0.7914      0.7203    134.62      88.36       96.86  0.0326   21.00%   13.27%   11474
```
