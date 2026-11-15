Reference run in MATLAB

1. Do not add the toolbox folder (toolbox/optim, with fmincon.m, ga.m, MultiStart.m,
   optimoptions.m, ...) to the MATLAB path. record_reference checks this and adds only
   ../tests, where the test scripts are.
2. In MATLAB:  cd <this folder>;  record_reference
3. The run writes reference_matlab.txt (plus out_lp6.csv and out_qp2.csv) here.

Needs the Optimization Toolbox; test_global also needs the Global Optimization Toolbox
(it is skipped if that toolbox is not licensed). test_global takes a few minutes: it runs
particleswarm and ga 20 times on each of 3 test functions.

The reference_matlab.txt in this folder was recorded with MATLAB R2026b (the only release
available to the author), not R2023b. See ../README.md, "Comparison with MATLAB".
