# Held-out execution scope

Recorded after the 768/16,384 validation sweep and before any new test-view fields. The original promotion order and gates in PROTOCOL.md remain fixed.

Use all 70,652 observations for the primary `test_side` quality and runtime comparison. Use 16,384 observations for each secondary held-out view group (`test_back`, `test_other_side`), with full float64 factors for these inexpensive generalization checks. Secondary results do not establish large-N performance on those groups. Evaluate the old setting and the frozen promoted setting in each case.

The primary quality run also evaluates the original 128 benchmark query indices. Reuse those independently refined full-kernel moments as the timing oracle. Charge the competitors their own fresh setup, contacts, and all 128 outputs; exclude reference construction and display work. Test both plain float32 Cholesky and residual-corrected mixed Cholesky. Repeat the strongest passing dense recipe and an accuracy-passing H² recipe three times at the primary size. If a method fails accuracy, retain the failed trial and adjust only its numerical solver settings, never the frozen physical model.

Use 80×96×80 and 120×144×120 grids for matched old/new mean-surface distances and display convergence. These grids discretize the display, not the free-space GP. Render the actual computed meshes. No test-derived change to model settings is permitted.
