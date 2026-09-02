import subprocess

## GPU specification registry                                        ##
## Same plain-dict style as the nvp/ncu metric dicts in roofline_tool ##
##                                                                    ##
## Each entry holds the ceilings for both roofline modes:             ##
##                                                                    ##
##  Instruction roofline (Ding & Williams, LBL 2019):                 ##
##    peak_gips : theoretical peak warp instructions/s                ##
##                ( = sm * 4 warp schedulers * boost_ghz )            ##
##    l1_gtxn / l2_gtxn / hbm_gtxn : cache/HBM ceilings in giga       ##
##                32-byte-transactions per second (GTXN/s)            ##
##                                                                    ##
##  Compute (FLOP) roofline:                                          ##
##    fp64..fp4 : dense peak TFLOP/s per precision (None if the       ##
##                format is not supported by the architecture)        ##
##    hbm_gbs   : peak HBM/DRAM bandwidth in GB/s                     ##
##                                                                    ##
## Notes on provenance:                                               ##
##  - V100S and A100 keep the exact instruction-roofline ceilings     ##
##    from the LBL paper / Ampere whitepaper (so the bundled example  ##
##    reproduces). A100 values are for the 40GB card used there.      ##
##  - For Hopper/Blackwell/RTX the paper did not publish L1/L2 GTXN/s ##
##    ceilings; they are left None (run GPU_Microbenchmarks to        ##
##    measure them). peak_gips is derived from the SM/clock formula   ##
##    and hbm_gtxn from hbm_gbs / 32.                                 ##
##  - Tensor TFLOP/s are DENSE (non-sparse) figures with FP32         ##
##    accumulate. Datasheet headline numbers are usually 2x these     ##
##    (2:4 structured sparsity); consumer Ada/Blackwell also run      ##
##    FP16/FP8 with FP16 accumulate at 2x the FP32-accumulate rate    ##
##    quoted here. H100/H200 verified vs the Hopper whitepaper; the   ##
##    RTX 4090/5090 rows are from the RTX Blackwell whitepaper        ##
##    (Appendix A). B200/GB200 non-tensor FP32 is not published in a  ##
##    primary whitepaper -- treat it as approximate.                  ##

WARP_SCHEDULERS = 4  # warp schedulers per SM on all supported archs
BYTES_PER_TRANSACTION = 32  # a memory transaction is a 32-byte sector


def warp_gips(sm, boost_ghz):
    # Theoretical peak warp instruction issue rate (instruction roofline).
    return sm * WARP_SCHEDULERS * boost_ghz


SPECS = {

    "V100S": {
        "arch": "Volta", "sm": 80, "boost_ghz": 1.53,
        # instruction roofline (LBL paper, Tesla V100)
        "peak_gips": 489.6, "l1_gtxn": 437.5, "l2_gtxn": 93.6, "hbm_gtxn": 25.9,
        # compute roofline (V100S datasheet)
        "fp64": 8.2, "fp32": 16.4, "tf32": None,
        "fp16": 130, "bf16": None, "fp8": None, "fp4": None,
        "hbm_gbs": 1134,
    },

    "A100": {
        "arch": "Ampere", "sm": 108, "boost_ghz": 1.41,
        # instruction roofline (Ampere whitepaper) -> 108*4*1.41 = 609.12.
        # L1/L2 GTXN/s from the LBL microbenchmarks; HBM from the 80GB card.
        "peak_gips": 609.12, "l1_gtxn": 1312.5, "l2_gtxn": 215.3,
        "hbm_gtxn": 2039 / BYTES_PER_TRANSACTION,
        # compute roofline (A100 80GB datasheet, dense; HBM2e = 2039 GB/s)
        "fp64": 9.7, "fp32": 19.5, "tf32": 156,
        "fp16": 312, "bf16": 312, "fp8": None, "fp4": None,
        "hbm_gbs": 2039,
    },

    "H100-SXM": {
        "arch": "Hopper", "sm": 132, "boost_ghz": 1.98,
        # instruction roofline: peak_gips derived; L1/L2 need microbenchmarking
        "peak_gips": warp_gips(132, 1.98), "l1_gtxn": None, "l2_gtxn": None,
        "hbm_gtxn": 3350 / BYTES_PER_TRANSACTION,
        # compute roofline (Hopper whitepaper, HBM3, dense)
        "fp64": 34, "fp32": 67, "tf32": 494,
        "fp16": 989, "bf16": 989, "fp8": 1979, "fp4": None,
        "hbm_gbs": 3350,
    },

    "H200-SXM": {
        "arch": "Hopper", "sm": 132, "boost_ghz": 1.98,
        "peak_gips": warp_gips(132, 1.98), "l1_gtxn": None, "l2_gtxn": None,
        "hbm_gtxn": 4800 / BYTES_PER_TRANSACTION,
        # same compute as H100, HBM3e -> 4.8 TB/s
        "fp64": 34, "fp32": 67, "tf32": 494,
        "fp16": 989, "bf16": 989, "fp8": 1979, "fp4": None,
        "hbm_gbs": 4800,
    },

    "B200": {
        "arch": "Blackwell", "sm": 148, "boost_ghz": 1.86,
        "peak_gips": warp_gips(148, 1.86), "l1_gtxn": None, "l2_gtxn": None,
        "hbm_gtxn": 8000 / BYTES_PER_TRANSACTION,
        # compute roofline (Blackwell B200 datasheet, dense; adds FP4).
        # fp32 non-tensor is a third-party figure (no primary whitepaper).
        "fp64": 37, "fp32": 75, "tf32": 1100,
        "fp16": 2250, "bf16": 2250, "fp8": 4500, "fp4": 9000,
        "hbm_gbs": 8000,
    },

    "GB200": {
        "arch": "Blackwell", "sm": 148, "boost_ghz": 1.86,
        # per-GPU figures of the Grace-Blackwell superchip (one of two B200 dies-pair)
        "peak_gips": warp_gips(148, 1.86), "l1_gtxn": None, "l2_gtxn": None,
        "hbm_gtxn": 8000 / BYTES_PER_TRANSACTION,
        "fp64": 40, "fp32": 80, "tf32": 1250,
        "fp16": 2500, "bf16": 2500, "fp8": 5000, "fp4": 10000,
        "hbm_gbs": 8000,
    },

    "RTX4090": {
        "arch": "Ada", "sm": 128, "boost_ghz": 2.52,
        "peak_gips": warp_gips(128, 2.52), "l1_gtxn": None, "l2_gtxn": None,
        "hbm_gtxn": 1008 / BYTES_PER_TRANSACTION,
        # compute roofline (RTX Blackwell whitepaper App. A, Ada column;
        # dense, FP32 accumulate; GDDR6X). FP16/FP8 accumulate double these.
        "fp64": 1.3, "fp32": 82.6, "tf32": 82.6,
        "fp16": 165.2, "bf16": 165.2, "fp8": 330.3, "fp4": None,
        "hbm_gbs": 1008,
    },

    "RTX5090": {
        "arch": "Blackwell", "sm": 170, "boost_ghz": 2.407,
        "peak_gips": warp_gips(170, 2.407), "l1_gtxn": None, "l2_gtxn": None,
        "hbm_gtxn": 1792 / BYTES_PER_TRANSACTION,
        # compute roofline (RTX Blackwell whitepaper App. A, GB202 column;
        # dense, FP32 accumulate; GDDR7, adds FP4). FP16/FP8 accumulate
        # double the fp16/fp8 rate (dense FP4 = 1676, 3352 with sparsity).
        "fp64": 1.6, "fp32": 104.8, "tf32": 104.8,
        "fp16": 209.5, "bf16": 209.5, "fp8": 419, "fp4": 1676,
        "hbm_gbs": 1792,
    },

}

## Precision keys usable for the compute roofline, in display order ##
PRECISIONS = ["fp64", "fp32", "tf32", "fp16", "bf16", "fp8", "fp4"]


def list_gpus():
    return list(SPECS.keys())


def get_spec(name):
    if name not in SPECS:
        raise KeyError(
            f"Unknown GPU '{name}'. Available: {', '.join(list_gpus())}")
    return SPECS[name]


def autodetect():
    # Query the installed GPU via nvidia-smi and map it to a registry key.
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"Could not run nvidia-smi for --autodetect ({e}). "
                         f"Pass --gpu explicitly, one of: {', '.join(list_gpus())}")
    lines = query.stdout.strip().splitlines()
    if not lines:
        # nvidia-smi returned 0 with no output: driver present but no GPU
        # visible (empty container, --gpus none, etc.). Match the friendly
        # SystemExit the FileNotFoundError / CalledProcessError branch raises.
        raise SystemExit("nvidia-smi returned no GPU name for --autodetect. "
                         f"Pass --gpu explicitly, one of: {', '.join(list_gpus())}")
    return match_name(lines[0])


def match_name(name):
    # Map a free-form GPU name (e.g. "NVIDIA H100 80GB HBM3") to a SPECS key.
    upper = name.upper().replace(" ", "").replace("-", "")
    aliases = {
        "V100": "V100S", "A100": "A100",
        "H100": "H100-SXM", "H200": "H200-SXM",
        "GB200": "GB200", "B200": "B200",
        "RTX5090": "RTX5090", "5090": "RTX5090",
        "RTX4090": "RTX4090", "4090": "RTX4090",
    }
    for token, key in aliases.items():
        if token in upper:
            return key
    raise KeyError(
        f"Could not match '{name}' to a known GPU. Available: "
        f"{', '.join(list_gpus())}")
