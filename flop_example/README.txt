Illustrative (synthetic) data for the compute/FLOP roofline mode.

These CSVs mimic the columns produced by `ncu --csv` when collecting the
FLOP-counter + dram__bytes metrics added to ../multi_kernels/profile_ncu.sh.
They are hand-authored, not a real profile, and exist only so the FLOP
roofline can be demonstrated without a GPU:

  saxpy : memory-bound  (low arithmetic intensity, sits on the HBM ceiling)
  gemm  : compute-bound (high arithmetic intensity, near the FP32 ceiling)

Reproduce the plot with:

  python roofline_tool.py --gpu A100 --mode flop --profiler ncu \
    --timing flop_example/timing.csv --metrics flop_example/metrics.csv \
    --kernels saxpy gemm --precisions fp32 fp16 \
    --out flop_example/roofline_flop.png
