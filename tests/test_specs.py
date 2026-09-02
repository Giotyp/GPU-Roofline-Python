import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gpu_specs


def test_a100_peak_gips_matches_formula():
    # 108 SMs * 4 warp schedulers * 1.41 GHz = 609.12 GIPS
    spec = gpu_specs.get_spec("A100")
    assert spec["peak_gips"] == 609.12
    assert gpu_specs.warp_gips(spec["sm"], spec["boost_ghz"]) == spec["peak_gips"]


def test_derived_hbm_gtxn_matches_bandwidth():
    # For GPUs whose HBM ceiling is derived from GB/s, hbm_gtxn * 32 == hbm_gbs.
    for name in ["A100", "H100-SXM", "H200-SXM", "B200", "RTX4090", "RTX5090"]:
        spec = gpu_specs.get_spec(name)
        assert abs(spec["hbm_gtxn"] * gpu_specs.BYTES_PER_TRANSACTION
                   - spec["hbm_gbs"]) < 1e-6, name


def test_every_spec_has_required_keys():
    required = ["arch", "sm", "boost_ghz", "peak_gips", "hbm_gbs"] + gpu_specs.PRECISIONS
    for name, spec in gpu_specs.SPECS.items():
        for key in required:
            assert key in spec, f"{name} missing {key}"


def test_unknown_gpu_raises():
    try:
        gpu_specs.get_spec("NOTAGPU")
    except KeyError:
        return
    assert False, "expected KeyError for unknown GPU"


def test_match_name_aliases():
    assert gpu_specs.match_name("NVIDIA H100 80GB HBM3") == "H100-SXM"
    assert gpu_specs.match_name("NVIDIA GeForce RTX 5090") == "RTX5090"
    assert gpu_specs.match_name("Tesla V100S-PCIE-32GB") == "V100S"


def test_autodetect_exits_cleanly_when_nvidia_smi_returns_empty_stdout():
    # nvidia-smi exits 0 with no output when the driver is present but no
    # GPU is visible (empty container, --gpus none, etc.). Instead of an
    # uncaught IndexError, autodetect should raise SystemExit with the same
    # "pass --gpu explicitly" message the FileNotFoundError branch uses.
    import subprocess as real_subprocess
    original_run = gpu_specs.subprocess.run

    class _EmptyCompleted:
        stdout = "   \n"
        returncode = 0

    gpu_specs.subprocess.run = lambda *args, **kwargs: _EmptyCompleted()
    try:
        try:
            gpu_specs.autodetect()
        except SystemExit as exc:
            assert "no GPU name" in str(exc)
            return
        assert False, "expected SystemExit for empty nvidia-smi output"
    finally:
        gpu_specs.subprocess.run = original_run
        assert gpu_specs.subprocess is real_subprocess


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
