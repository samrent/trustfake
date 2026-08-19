# WP1 condition table -- vit_large_patch14_clip_224

- split: `test`, n = 15316
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- temperature 0.9088 fitted on `predictions_vit_large_patch14_clip_224_calib_squarecrop` and frozen
- AURC/AUGRC convention: tie blocks collapsed, block-size-weighted mean over operating points
- ECE column: 15 bins, equal-mass
- **TRIVIAL BASELINE on this split: 'width==height -> fake' = 0.9785 accuracy (majority class 0.5000). Read every model accuracy against that number.**

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
            squarecrop  15316  0.9291  0.9288     0.9802      0.8902     11.75       9.75        9.17  0.0041    1.96%    0.38%   15265
      squarecrop (T=1)  15316  0.9291  0.9288     0.9802      0.8902     11.75       9.75        9.17  0.0066    1.96%    0.38%   15265
```

