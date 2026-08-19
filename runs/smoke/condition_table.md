# WP1 condition table -- tf_efficientnet_b0

- split: `test`, n = 256
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- temperature 1.2237 fitted on `predictions_tf_efficientnet_b0_calib_clean` and frozen
- AURC/AUGRC convention: tie blocks collapsed, block-size-weighted mean over operating points
- ECE column: 15 bins, equal-mass

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
                 clean    256  0.7734  0.7752     0.8589      0.7471     99.36      70.42       71.06  0.0565   17.16%   10.94%     256
           clean (T=1)    256  0.7734  0.7752     0.8589      0.7471     99.36      70.42       71.06  0.0673   17.16%   10.94%     256
          ace_eps0.002    256  0.7734  0.7752     0.5990      0.0022    559.78     200.96      531.48  0.5164   28.43%   45.31%     256
    ace_eps0.002 (T=1)    256  0.7734  0.7752     0.5990      0.0022    559.78     200.96      531.48  0.5104   28.43%   45.31%     256
```

clean, broken out by fake class:

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
     real vs synthetic    187  0.8128  0.7552     0.9203      0.7810     69.51      51.33       50.28  0.0775   13.42%    7.53%     187
      real vs tampered    194  0.7423  0.6479     0.8037      0.7029    135.26      90.71       98.10  0.0551   21.29%   13.40%     194
```
