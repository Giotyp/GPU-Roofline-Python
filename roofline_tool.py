import argparse
import io
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gpu_specs

## Create dictionaries for ncu and nvprof  ##
## with appropriate metric names           ##

nvp = {
    "time_kernel": "Name",
    "Average": "Avg",
    "metric_kernel": "Kernel",
    "gld":"gld_transactions",
    "gst":"gst_transactions",
    "shld":"shared_load_transactions",
    "shst":"shared_store_transactions",
    "l2rd":"l2_read_transactions",
    "l2wr":"l2_write_transactions",
    "drrd":"dram_read_transactions",
    "drwr":"dram_write_transactions",
    # compute (FLOP) roofline
    "fadd":"flop_count_sp_add",
    "fmul":"flop_count_sp_mul",
    "ffma":"flop_count_sp_fma",
    "dram_bytes":"dram_read_bytes",  # nvprof: add read+write, see app_char_flop
    "dram_bytes_wr":"dram_write_bytes",
}

ncu = {
    "time_kernel": "Kernel Name",
    "Average": "Average",
    "metric_kernel": "Kernel Name",
    "gld":"l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "gst":"l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "shld":"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "shst":"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l2rd":"lts__t_sectors_op_read.sum",
    "l2wr":"lts__t_sectors_op_write.sum",
    "l2at":"lts__t_sectors_op_atom.sum",
    "l2red":"lts__t_sectors_op_red.sum",
    "drrd":"dram__sectors_read.sum",
    "drwr":"dram__sectors_write.sum",
    # compute (FLOP) roofline
    "fadd":"sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "fmul":"sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "ffma":"sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "dram_bytes":"dram__bytes.sum",
    "dram_bytes_wr":None,  # ncu dram__bytes.sum already counts read+write
}


# Function to return float value from row object
# ncu stores numbers in ',' format e.g. 750,000.0
def float_val(x):
    return float(x.to_string().split(' ')[-1].replace(',',''))


# Read a profiler CSV, stripping nvprof banner lines (e.g.
# "==5328== NVPROF is profiling process 5328, command: ./transpose")
# WITHOUT mutating the file on disk. Returns a parsed DataFrame.
def load_clean_csv(path):
    with open(path, "r") as f:
        lines = [line for line in f if "==" not in line]
    return pd.read_csv(io.StringIO("".join(lines)))


def create_instruction_graph(spec, memories):
    ## Instruction-roofline ceilings, pulled from the GPU spec ##

    peak = spec["peak_gips"]
    l1_bw = spec["l1_gtxn"]
    l2_bw = spec["l2_gtxn"]
    hbm_bw = spec["hbm_gtxn"]


    ## Plotting ##

    # Modify specified figure parameters according to needs

    fig = plt.figure(figsize=(8,4))
    # (left,bottom,width,height) of the figure
    ax = plt.axes((0.1,0.1,0.8,0.8))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Instruction Intensity (Warp Instructions per Transaction)')
    ax.set_ylabel('Performance (warp GIPS)')

    xmin, xmax, ymin, ymax = -2, 2, -1, 3

    ax.set_xlim(10**xmin, 10**xmax)
    ax.set_ylim(10**ymin, 10**ymax)

    instr_min = 10**xmin

    peak_x = np.asarray([peak/l1_bw if l1_bw else instr_min, 10**xmax]) # performance ceiling
    peak_y = np.asarray([peak, peak])

    l1, l2, hbm = memories

    ax.plot(peak_x, peak_y, color='0') # Performance ceiling

    # Each cache/HBM ceiling is only drawn when the spec provides it
    # (Hopper/Blackwell/RTX L1/L2 need microbenchmarking, see gpu_specs).
    if l1 and l1_bw:
        l1_x = np.asarray([instr_min, peak/l1_bw])
        l1_y = np.asarray([l1_bw*instr_min, peak])
        ax.plot(l1_x, l1_y, color='r', label=f'L1 {l1_bw} GTXN/s') # L1 ceiling
    if l2 and l2_bw:
        l2_x = np.asarray([instr_min, peak/l2_bw])
        l2_y = np.asarray([l2_bw*instr_min, peak])
        ax.plot(l2_x, l2_y, color='g', label=f'L2 {l2_bw} GTXN/s') # L2 ceiling
    if hbm and hbm_bw:
        hbm_x = np.asarray([instr_min, peak/hbm_bw])
        hbm_y = np.asarray([hbm_bw*instr_min, peak])
        ax.plot(hbm_x, hbm_y, color='b', label=f'HBM {round(hbm_bw,1)} GTXN/s') # HBM ceiling

    # text for peak performance
    elbow = peak/l1_bw if l1_bw else instr_min
    ax.text(elbow, peak+100, f'Theoretical Peak: {round(peak,1)} warp GIPS')

    return ax,fig


def create_flop_graph(spec, precisions):
    ## Classic compute (FLOP) roofline ##
    ## X = arithmetic intensity (FLOP/byte), Y = attainable TFLOP/s ##

    hbm_bw = spec["hbm_gbs"] / 1000.0  # GB/s -> TB/s (= TFLOP per FLOP/byte)

    fig = plt.figure(figsize=(8,4))
    ax = plt.axes((0.1,0.1,0.8,0.8))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOP / Byte)')
    ax.set_ylabel('Performance (TFLOP/s)')

    xmin, xmax = -2, 4
    ax.set_xlim(10**xmin, 10**xmax)

    ai_min = 10**xmin
    ai_max = 10**xmax

    # one horizontal compute ceiling per requested precision
    peaks = [spec[p] for p in precisions if spec.get(p)]
    ymax_peak = max(peaks) if peaks else 1
    ax.set_ylim(10**-1, ymax_peak*2)

    colours = itertools.cycle(['r','g','b','m','c','y','k'])
    for p in precisions:
        peak = spec.get(p)
        if not peak:
            continue  # precision unsupported on this GPU
        colour = next(colours)
        elbow = peak/hbm_bw  # ridge point (FLOP/byte)
        ax.plot([elbow, ai_max], [peak, peak], color=colour,
                label=f'{p.upper()} {peak} TFLOP/s')  # compute ceiling

    # single diagonal HBM memory ceiling
    ax.plot([ai_min, ai_max], [hbm_bw*ai_min, hbm_bw*ai_max], color='0',
            label=f'HBM {spec["hbm_gbs"]} GB/s')

    return ax,fig



def timing(kernel_stats, kernel_name, time_file, profiler):
    timing = load_clean_csv(time_file)

    ## Kernel Time ##
    time_row = timing.loc[(timing[profiler["time_kernel"]] == kernel_name)]

    if time_row.empty:
        names = timing[profiler["time_kernel"]].dropna().unique().tolist()
        raise SystemExit(f"Kernel '{kernel_name}' not found in {time_file}. "
                         f"Available: {names}")

    kernel_time = time_row[profiler["Average"]] # average kernel time
    kernel_time = float_val(kernel_time)

    if profiler == nvp:
        unit_row = timing.loc[(timing['Time(%)'].str.match('%', na=False))]
        unit = unit_row.iloc[0]["Avg"] # unit used
        unit += "econd" # convert to ncu display
    elif profiler == ncu:
        unit = time_row["Metric Unit"].to_list()[0] # unit used

    # change to usecond
    if unit == 'msecond':
        kernel_time *= 1000
    elif unit == 'nsecond':
        kernel_time /= 1000
    elif unit == 'second':
        kernel_time *= 1000000

    kernel_stats['kernel_time'] = kernel_time



def find_inst(kernel_stats, kernel_name, events_file, profiler):
    events = load_clean_csv(events_file)

    # Total Instructions
    instructions = events.loc[(events[profiler["metric_kernel"]] == kernel_name)]
    kernel_inst = instructions[profiler["Average"]]
    kernel_inst = float_val(kernel_inst)
    total_inst_nrml = kernel_inst

    total_inst_nrml /= 32

    kernel_stats['total_inst'] = total_inst_nrml



def app_char(kernel_stats, kernel_name, metrics_file, graph, memories, profiler, labels, colors, markers, mode):
    metrics = load_clean_csv(metrics_file)
    kernel_metrics = metrics.loc[(metrics[profiler["metric_kernel"]] == kernel_name)]

    l1, l2, hbm = memories

    if l1:
        ## L1 stats ##

        gld_stats = kernel_metrics.loc[kernel_metrics['Metric Name'] == profiler["gld"]]
        gld_trans = int(float_val(gld_stats[profiler["Average"]]))

        gst_stats = kernel_metrics.loc[(kernel_metrics["Metric Name"] == profiler["gst"])]
        gst_trans = int(float_val(gst_stats[profiler["Average"]]))

        sld_stats = kernel_metrics.loc[(kernel_metrics["Metric Name"] == profiler["shld"])]
        sld_trans = int(float_val(sld_stats[profiler["Average"]]))

        sst_stats = kernel_metrics.loc[(kernel_metrics["Metric Name"] == profiler["shst"])]
        sst_trans = int(float_val(sst_stats[profiler["Average"]]))

        l1_total = gld_trans + gst_trans + sld_trans + sst_trans

        l1_intensity = kernel_stats['total_inst'] / l1_total
        l1_performance = kernel_stats['total_inst'] / (1000 * kernel_stats['kernel_time']) # performance in  GIPS ( kernel_time in μsecs )


        if mode == 0:
            color = colors["l1"]
            marker = markers["l1"]
            label = labels["l1"]
        elif mode == 1:
            color = colors[kernel_name]
            marker = markers[kernel_name]
            label = labels[kernel_name]

        graph.plot(l1_intensity, l1_performance, color=color, marker = marker, label=label)


    if l2:
        ## L2 stats ##

        l2_at, l2_red = 0,0
        if profiler == ncu:
            l2_at = kernel_metrics.loc[(kernel_metrics["Metric Name"] == profiler["l2at"])]
            l2_at = int(float_val(l2_at[profiler["Average"]]))
            l2_red = kernel_metrics.loc[(kernel_metrics["Metric Name"] == profiler["l2red"])]
            l2_red = int(float_val(l2_red[profiler["Average"]]))

        l2_rd = kernel_metrics.loc[(kernel_metrics["Metric Name"] == profiler["l2rd"])]
        l2_rd = int(float_val(l2_rd[profiler["Average"]]))

        l2_read_trans =  l2_rd + l2_red + l2_at

        l2_wr = kernel_metrics.loc[(kernel_metrics["Metric Name"] == profiler["l2wr"])]
        l2_wr = int(float_val(l2_wr[profiler["Average"]]))

        l2_write_trans = l2_wr + l2_red + l2_at

        l2_total = l2_read_trans + l2_write_trans

        l2_intensity = kernel_stats['total_inst'] / l2_total
        l2_performance = kernel_stats['total_inst'] / (1000 * kernel_stats['kernel_time']) # performance in  GIPS ( kernel_time in μsecs )


        if mode == 0:
            color = colors["l2"]
            marker = markers["l2"]
            label = labels["l2"]
        elif mode == 1:
            color = colors[kernel_name]
            marker = markers[kernel_name]
            label = labels[kernel_name]

        graph.plot(l2_intensity, l2_performance, color=color, marker = marker, label=label)


    if hbm:
    ## HBM stats ##

        dram_rd = kernel_metrics.loc[kernel_metrics['Metric Name'] == profiler["drrd"]]
        dram_rd = int(float_val(dram_rd[profiler["Average"]]))

        dram_wr = kernel_metrics.loc[kernel_metrics['Metric Name'] == profiler["drwr"]]
        dram_wr = int(float_val(dram_wr[profiler["Average"]]))

        dram_total = dram_rd + dram_wr

        hbm_intensity = kernel_stats['total_inst'] / dram_total
        hbm_performance = kernel_stats['total_inst'] / (1000 * kernel_stats['kernel_time']) # performance in  GIPS ( kernel_time in μsecs )


        if mode == 0:
            color = colors["hbm"]
            marker = markers["hbm"]
            label = labels["hbm"]
        elif mode == 1:
            color = colors[kernel_name]
            marker = markers[kernel_name]
            label = labels[kernel_name]

        graph.plot(hbm_intensity, hbm_performance, color=color, marker = marker, label=label)



def app_char_flop(kernel_stats, kernel_name, metrics_file, graph, profiler, label, color, marker):
    ## Characterize a kernel on the compute (FLOP) roofline ##

    metrics = load_clean_csv(metrics_file)
    kernel_metrics = metrics.loc[(metrics[profiler["metric_kernel"]] == kernel_name)]

    def metric(key):
        row = kernel_metrics.loc[kernel_metrics['Metric Name'] == profiler[key]]
        return float_val(row[profiler["Average"]])

    # single-precision FLOPs: add + mul + 2*fma
    flops = metric("fadd") + metric("fmul") + 2 * metric("ffma")

    # DRAM bytes moved (ncu counts read+write; nvprof needs both summed)
    dram_bytes = metric("dram_bytes")
    if profiler["dram_bytes_wr"] is not None:
        dram_bytes += metric("dram_bytes_wr")

    intensity = flops / dram_bytes  # FLOP / byte
    # TFLOP/s: kernel_time is in microseconds
    performance = flops / (kernel_stats['kernel_time'] * 1e6)

    graph.plot(intensity, performance, color=color, marker=marker, label=label)



def build_style(keys):
    ## Auto-assign colors/markers/labels for arbitrary kernel names ##
    palette = itertools.cycle(['y','m','c','r','g','b','k'])
    shapes = itertools.cycle(['s','o','^','D','v','P','X'])
    colors = {k: next(palette) for k in keys}
    markers = {k: next(shapes) for k in keys}
    labels = {k: k for k in keys}
    return colors, markers, labels


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot instruction or FLOP roofline models for GPU kernels.")

    gpu = p.add_mutually_exclusive_group(required=True)
    gpu.add_argument("--gpu", choices=gpu_specs.list_gpus(),
                     help="GPU spec to use for the ceilings")
    gpu.add_argument("--autodetect", action="store_true",
                     help="detect the installed GPU via nvidia-smi")

    p.add_argument("--mode", choices=["instruction", "flop"],
                   default="instruction", help="which roofline model to draw")
    p.add_argument("--roofline-type", choices=["hierarchical", "multi"],
                   default="multi",
                   help="hierarchical: single kernel, L1/L2/HBM points; "
                        "multi: several kernels on chosen memories")
    p.add_argument("--profiler", choices=["ncu", "nvprof"], default="ncu")

    p.add_argument("--timing", required=True, help="timing csv")
    p.add_argument("--events", help="events csv (instruction mode)")
    p.add_argument("--metrics", required=True, help="metrics csv")

    p.add_argument("--kernels", nargs="+", required=True,
                   help="kernel name(s) as they appear in the csv files")

    p.add_argument("--memories", nargs="+", default=["l1", "l2", "hbm"],
                   choices=["l1", "l2", "hbm"],
                   help="instruction mode: memory ceilings / points")
    p.add_argument("--precisions", nargs="+", default=["fp16", "fp8"],
                   choices=gpu_specs.PRECISIONS,
                   help="flop mode: precision compute ceilings")

    p.add_argument("--out", default="roofline.png", help="output png path")
    p.add_argument("--title", help="plot title")
    return p.parse_args()


def main():
    args = parse_args()

    gpu_name = gpu_specs.autodetect() if args.autodetect else args.gpu
    spec = gpu_specs.get_spec(gpu_name)
    profiler = ncu if args.profiler == "ncu" else nvp

    if args.mode == "flop":
        ax, fig = create_flop_graph(spec, args.precisions)
        colors, markers, labels = build_style(args.kernels)
        for kernel_name in args.kernels:
            kernel_stats = {}
            timing(kernel_stats, kernel_name, args.timing, profiler)
            app_char_flop(kernel_stats, kernel_name, args.metrics, ax, profiler,
                          labels[kernel_name], colors[kernel_name], markers[kernel_name])
        default_title = f"FLOP Roofline (NVIDIA {gpu_name})"
    else:
        mode = 0 if args.roofline_type == "hierarchical" else 1
        # ceilings: draw every memory level the spec supports
        # points: only the memory levels the user selected via --memories
        memories_ceil = [True, True, True]
        memories_plot = ["l1" in args.memories, "l2" in args.memories, "hbm" in args.memories]
        # hierarchical: one kernel plotted at L1/L2/HBM; use l1/l2/hbm style keys
        style_keys = ["l1", "l2", "hbm"] if mode == 0 else args.kernels
        colors, markers, labels = build_style(style_keys)

        ax, fig = create_instruction_graph(spec, memories_ceil)
        for kernel_name in args.kernels:
            kernel_stats = {}
            timing(kernel_stats, kernel_name, args.timing, profiler)
            find_inst(kernel_stats, kernel_name, args.events, profiler)
            app_char(kernel_stats, kernel_name, args.metrics, ax, memories_plot,
                     profiler, labels, colors, markers, mode)
        default_title = f"Instruction Roofline (NVIDIA {gpu_name})"

    ax.legend(loc='lower right', fontsize='8')
    ax.set_title(args.title or default_title)
    ax.grid(True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
