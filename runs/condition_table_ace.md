# WP1 condition table -- tf_efficientnet_b0

- split: `test`, n = 15316
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- temperature 1.2785 fitted on `predictions_tf_efficientnet_b0_calib_clean` and frozen
- AURC/AUGRC convention: tie blocks collapsed, block-size-weighted mean over operating points
- ECE column: 15 bins, equal-mass

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
                 clean  15316  0.8311  0.8303     0.9065      0.7703     67.05      46.52       51.91  0.0080   11.25%    5.77%   15264
           clean (T=1)  15316  0.8311  0.8303     0.9065      0.7703     67.05      46.52       51.91  0.0318   11.25%    5.77%   15264
         ace_eps0.0005  15316  0.8311  0.8303     0.7270      0.1295    353.83     136.47      338.68  0.3257   21.11%   32.76%   15264
   ace_eps0.0005 (T=1)  15316  0.8311  0.8303     0.7270      0.1295    353.83     136.47      338.68  0.3252   21.11%   32.76%   15264
          ace_eps0.002  15316  0.8311  0.8303     0.6915      0.0030    466.86     154.23      451.71  0.4732   21.11%   33.77%   15276
    ace_eps0.002 (T=1)  15316  0.8311  0.8303     0.6915      0.0030    466.86     154.23      451.71  0.4565   21.11%   33.77%   15276
          ace_eps0.005  15316  0.8311  0.8303     0.6907      0.0002    469.14     154.63      453.99  0.4876   21.11%   33.78%   15278
    ace_eps0.005 (T=1)  15316  0.8311  0.8303     0.6907      0.0002    469.14     154.63      453.99  0.4715   21.11%   33.78%   15278
```

clean, broken out by fake class:

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
     real vs synthetic  11487  0.8763  0.8376     0.9697      0.8242     35.56      26.72       27.56  0.0312    6.52%    2.25%   11452
      real vs tampered  11487  0.7892  0.6874     0.8433      0.7230    106.30      68.33       82.31  0.0208   16.03%   10.20%   11469
```
