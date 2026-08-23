# WP1 condition table -- vit_l14_e2e

- split: `test`, n = 4000
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- temperature 0.9107 fitted on `predictions_vit_l14_e2e_calib_clean` and frozen
- AURC/AUGRC convention: tie blocks collapsed, block-size-weighted mean over operating points
- ECE column: 15 bins, equal-mass
- **TRIVIAL BASELINE on this split: 'width==height -> fake' = 0.9785 accuracy (majority class 0.5000). Read every model accuracy against that number.**

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
                 clean   4000  0.9323  0.9322     0.9798      0.8779     12.09      10.01        9.73  0.0128    2.12%    0.40%    3997
           clean (T=1)   4000  0.9323  0.9322     0.9798      0.8779     12.09      10.01        9.73  0.0132    2.12%    0.40%    3997
    ace_uint8_eps0.005   4000  0.9323  0.9322     0.8932      0.1812    147.49      54.02      145.13  0.1569    8.41%   12.35%    3998
ace_uint8_eps0.005 (T=1)   4000  0.9323  0.9322     0.8932      0.1812    147.49      54.02      145.13  0.1621    8.41%   12.35%    3998
          ace_eps0.005   4000  0.9323  0.9322     0.8703      0.0102    242.25      64.82      239.89  0.3419    8.47%   13.55%    3998
    ace_eps0.005 (T=1)   4000  0.9323  0.9322     0.8703      0.0102    242.25      64.82      239.89  0.3518    8.47%   13.55%    3998
```

clean, broken out by fake class:

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
     real vs synthetic   2986  0.9514  0.9308     0.9962      0.9128      5.99       5.22        4.78  0.0095    1.01%    0.13%    2983
      real vs tampered   3014  0.9137  0.8723     0.9638      0.8463     20.05      15.85       16.21  0.0136    3.82%    0.93%    3014
```
