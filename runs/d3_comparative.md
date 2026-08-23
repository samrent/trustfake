# WP3 / D3 — comparative results

All three arms: one shared initialisation (`phase1_init.pt`), 12 epochs, identical data
protocol and identical evaluation. Differences are of METHOD, not of budget.
Each model's temperature is fitted on its OWN clean calib split and frozen before any
attacked row is scored.

```
TRIVIAL BASELINE on this split: 'width==height -> fake' = 0.9785 accuracy (majority class 0.5000). Read every model accuracy against that number.

condition                                                           standard_eps2_255                  at_pgd_eps2_255                  trades_eps2_255
-------------------------------------------------------------------------------------------------------------------------------------------------------
clean                                                0.8435   0.7945     52.5  15270 0.8177   0.8006     62.3  15270 0.7877   0.7761     82.5  15271
realistic: JPEG q50                                  0.7198   0.7188    140.4  15269 0.8169   0.7997     62.8  15267 0.7876   0.7730     83.4  15266
realistic: 0.5x resize                               0.7987   0.7793     75.7  15267 0.7857   0.7853     80.6  15271 0.7649   0.7586    100.9  15270
adv: PGD eps=2/255 (=AT radius; label)               0.0000      nan   1000.0  10102 0.7192   0.7054    146.8  15315 0.7062   0.6799    170.3  15315
adv: ACE uint8 cap 1.3/255 (conf; eff varies by model) 0.8435   0.1093    329.5  15282 0.8177   0.6659     94.5  15268 0.7877   0.6541    117.9  15270
adv: overconf eps=4/255 (2x AT radius; label-free)   0.8435   0.4135    194.2  12674 0.8177   0.7770     68.8  15260 0.7877   0.7789     81.3  15266

cells: accuracy | AUROC(failure) | AURC x10^-3 | n_op     T(standard_eps2_255)=3.338  T(at_pgd_eps2_255)=0.903  T(trades_eps2_255)=0.757
```
