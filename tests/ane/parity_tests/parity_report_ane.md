# ANE Parity Report: StatefulMambaHybrid1D

Generated: 2026-05-07 09:07:11
Config: d_model=512, d_state=64, headdim=64, num_heads=8, seq_length=224
Sequence: warmup=32 + measure=64 frames, seed=42, compute_units=CPU_AND_NE

## ANE: `pytorch FP32 (CPU)` → `CoreML CPU_AND_NE`
Tolerances: max_abs < 0.03, cosine_sim > 0.999

| Metric | Value | Status |
|--------|-------|--------|
| max_abs (worst frame)     | 8.556e-02    | **FAIL** |
| mean_abs_avg              | 2.857e-02 |  |
| cosine_sim_min            | 0.053147 |  |

**Failed checks:**
- max_abs=8.56e-02 > 3.00e-02
- cosine_sim_min=0.053147 < 0.999

### max_abs distribution across measurement frames
```
[8.46e-02,8.47e-02)    2  ##
[8.47e-02,8.48e-02)    8  ##########
[8.48e-02,8.49e-02)   17  #######################
[8.49e-02,8.51e-02)   22  ##############################
[8.51e-02,8.52e-02)   10  #############
[8.52e-02,8.53e-02)    3  ####
[8.53e-02,8.54e-02)    1  #
[8.54e-02,8.56e-02)    1  #
```

