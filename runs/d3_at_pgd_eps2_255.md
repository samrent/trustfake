# WP1 condition table -- effb0_at_pgd_eps2_255

- split: `test`, n = 15316
- provenance: SID-Set validation split, held-out slice (official test split withheld)
- temperature 0.9035 fitted on `predictions_effb0_at_pgd_eps2_255_calib_clean` and frozen
- AURC/AUGRC convention: tie blocks collapsed, block-size-weighted mean over operating points
- ECE column: 15 bins, equal-mass
- **TRIVIAL BASELINE on this split: 'width==height -> fake' = 0.9785 accuracy (majority class 0.5000). Read every model accuracy against that number.**

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
                 clean  15316  0.8177  0.7991     0.9046      0.8006     62.28      46.35       44.55  0.0184   11.78%    4.79%   15270
           clean (T=1)  15316  0.8177  0.7991     0.9046      0.8006     62.28      46.35       44.55  0.0267   11.78%    4.79%   15270
              jpeg_q50  15316  0.8169  0.7977     0.9039      0.7997     62.84      46.74       44.93  0.0190   11.87%    4.86%   15267
        jpeg_q50 (T=1)  15316  0.8169  0.7977     0.9039      0.7997     62.84      46.74       44.93  0.0267   11.87%    4.86%   15267
         downscale_0.5  15316  0.7857  0.7446     0.8848      0.7853     80.64      59.12       55.83  0.0227   15.06%    6.87%   15271
   downscale_0.5 (T=1)  15316  0.7857  0.7446     0.8848      0.7853     80.64      59.12       55.83  0.0222   15.06%    6.87%   15271
pgd_linf_eps0.00784_s10  15316  0.7192  0.6905     0.7863      0.7054    146.82      98.92      103.09  0.0619   23.87%   16.24%   15315
pgd_linf_eps0.00784_s10 (T=1)  15316  0.7192  0.6905     0.7863      0.7054    146.82      98.92      103.09  0.0502   23.87%   16.24%   15315
    ace_uint8_eps0.005  15316  0.8177  0.7991     0.8582      0.6659     94.49      66.42       76.76  0.0608   17.23%    9.31%   15268
ace_uint8_eps0.005 (T=1)  15316  0.8177  0.7991     0.8582      0.6659     94.49      66.42       76.76  0.0537   17.23%    9.31%   15268
overconf_eps0.0157_s20  15316  0.8177  0.7991     0.8971      0.7770     68.78      49.86       51.05  0.0954   12.38%    6.20%   15260
overconf_eps0.0157_s20 (T=1)  15316  0.8177  0.7991     0.8971      0.7770     68.78      49.86       51.05  0.0828   12.38%    6.20%   15267
```

clean, broken out by fake class:

```
             condition      n     acc      F1  AUROC_det  AUROC_fail  AURC_e-3  AUGRC_e-3  E-AURC_e-3     ECE  risk@.8  risk@.5    n_op
---------------------------------------------------------------------------------------------------------------------------------------
     real vs synthetic  11487  0.9310  0.9038     0.9868      0.8330     18.22      13.12       15.77  0.0887    3.01%    1.29%   11455
      real vs tampered  11487  0.7661  0.5767     0.8225      0.7319    117.47      75.41       87.68  0.0243   17.74%   10.88%   11473
```
