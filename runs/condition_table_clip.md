# WP1 condition table -- vit_large_patch14_clip_224

- split: `test`, n = 15316
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- temperature 0.9099 fitted on `predictions_vit_large_patch14_clip_224_calib_clean` and frozen
- AURC/AUGRC convention: tie blocks collapsed, block-size-weighted mean over operating points
- ECE column: 15 bins, equal-mass

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
                 clean  15316  0.9289  0.9285     0.9801      0.8898     11.83       9.81        9.23  0.0050    1.98%    0.40%   15268
           clean (T=1)  15316  0.9289  0.9285     0.9801      0.8898     11.83       9.81        9.23  0.0079    1.98%    0.40%   15268
```

clean, broken out by fake class:

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
     real vs synthetic  11487  0.9518  0.9317     0.9962      0.9215      5.47       4.77        4.28  0.0094    0.79%    0.09%   11454
      real vs tampered  11487  0.9095  0.8638     0.9640      0.8612     19.63      15.51       15.41  0.0088    3.56%    0.85%   11473
```
