import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

def create_graph():
    ## Define Bandwidths ##

    peak = 489.6 # theoretical peak 489.6 warp GIPS
    l1_bw = 437.5 # theoretical bandwidth of L1
    l1_elbow = peak/l1_bw

    l2_bw = 93.6 # theoretical bandwidth of L2
    l2_elbow = peak/l2_bw

    hbm_bw = 25.9 # theoretical bandwidth of HBM
    hbm_elbow = peak/hbm_bw


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
    instr_max = 10**xmax

    peak_x = np.asarray([l1_elbow, 10**xmax]) # performance ceiling
    peak_y = np.asarray([peak, peak]) 

    l1_x = np.asarray([instr_min, l1_elbow ]) # L1 ceiling
    l1_y = np.asarray([l1_bw*instr_min, peak])

    l2_x = np.asarray([instr_min, l2_elbow]) # L2 ceiling
    l2_y = np.asarray([l2_bw*instr_min, peak])

    hbm_x = np.asarray([instr_min, hbm_elbow]) # HBM ceiling
    hbm_y = np.asarray([hbm_bw*instr_min, peak])

    # add architectural characterization to figure
    ax.plot(peak_x, peak_y, color='0') # Performance ceiling
    ax.plot(l1_x, l1_y, color='r', label='L1 437.5 GTXN/s') # L1 ceiling
    ax.plot(l2_x, l2_y, color='g', label='L2 93.6 GTXN/s') # L2 ceiling
    ax.plot(hbm_x, hbm_y, color='b', label='HBM 25.9 GTXN/s') # HBM ceiling

    # text for peak performance
    ax.text(l1_elbow, peak+100, f'Theoretical Peak: {peak} warp GIPS')

    return ax,fig

def timing(kernel_dic, kernel_name, filename):
    # Format csv files to discard nvprof unwanted output
    # e.g. : ==5328== NVPROF is profiling process 5328, command: ./transpose

    with open(filename,"r+") as f:
        new_f = f.readlines()
        f.seek(0)
        for line in new_f:
            if "==" not in line:
                f.write(line)
        f.truncate()

    # get csv database with pandas
    timing = pd.read_csv(filename)

    ## Kernel Time ##
    time_row = timing.loc[(timing.Name.str.match(r'{}.'.format(kernel_name), na=False))]

    kernel_time = float(time_row["Avg"]) # average kernel time

    unit_row = timing.loc[(timing['Time(%)'].str.match('%', na=False))]
    unit = unit_row.iloc[0]["Avg"] # unit used

    # change to usec
    if unit == 'ms':
        kernel_time *= 1000
    elif unit == 'ns':
        kernel_time /= 1000
    elif unit == 's':
        kernel_time *= 1000000

    id = filename.split('.')[0].split('_')[1]
    kernel_dic[id] = {}
    kernel_dic[id]['kernel_time'] = kernel_time


def find_inst(kernel_dic, kernel_name, filename):
    with open(filename,"r+") as f:
        new_f = f.readlines()
        f.seek(0)
        for line in new_f:
            if "==" not in line:
                f.write(line)
        f.truncate()

    events = pd.read_csv(filename)

    # Total Instructions
    # If executing a python kernel there might be multiple calls 
    # to profiling function e.g. fft kernel
    instructions = events.loc[(events["Event Name"] == "thread_inst_executed")]
    kernel_inst = instructions.loc[(instructions.Kernel.str.match(r'{}.'.format(kernel_name), na=False))]
    total_inst_nrml = 0
    for inst in kernel_inst['Avg']:
        total_inst_nrml += int(inst)

    total_inst_nrml /= 32

    id = filename.split('.')[0].split('_')[1]
    kernel_dic[id]['total_inst'] = total_inst_nrml



def app_char(kernel_dic, kernel_name, filename, graph, simple):
    with open(filename,"r+") as f:
        new_f = f.readlines()
        f.seek(0)
        for line in new_f:
            if "==" not in line:
                f.write(line)
        f.truncate()

    metrics = pd.read_csv(filename)
    id = filename.split('.')[0].split('_')[1]

    ## L1 stats ##
    gld_stats = metrics.loc[(metrics["Metric Name"] == "gld_transactions")]
    gld_stats = gld_stats.loc[(gld_stats.Kernel.str.match(r'{}.'.format(kernel_name), na=False))]
    gld_transactions = 0
    for tr in gld_stats["Avg"]:
        gld_transactions += int(tr)

    gst_stats = metrics.loc[(metrics["Metric Name"] == "gst_transactions")]
    gst_stats = gst_stats.loc[(gst_stats.Kernel.str.match(r'{}.'.format(kernel_name), na=False))]
    gst_transactions = 0
    for tr in gst_stats["Avg"]:
        gst_transactions += int(tr)

    sld_stats = metrics.loc[(metrics["Metric Name"] == "shared_load_transactions")]
    sld_stats = sld_stats.loc[(sld_stats.Kernel.str.match(r'{}.'.format(kernel_name), na=False))]
    sld_transactions = 0
    for tr in sld_stats["Avg"]:
        sld_transactions += int(tr)

    sst_stats = metrics.loc[(metrics["Metric Name"] == "shared_store_transactions")]
    sst_stats = sst_stats.loc[(sst_stats.Kernel.str.match(r'{}.'.format(kernel_name), na=False))]
    sst_transactions = 0
    for tr in sst_stats["Avg"]:
        sst_transactions += int(tr)

    l1_total = gld_transactions + gst_transactions + sld_transactions + sst_transactions

    l1_intensity = kernel_dic[id]['total_inst'] / l1_total
    l1_performance = kernel_dic[id]['total_inst'] / (1000 * kernel_dic[id]['kernel_time']) # performance in  GIPS ( kernel_time in μsecs )

    if simple:
        graph.plot(l1_intensity, l1_performance, color='r', marker = 's', label="Total Instructions")
        return

    graph.plot(l1_intensity, l1_performance, color='r', marker = 's', label="L1 (tot_inst)")

    ## L2 stats ##

    l2_read_stats = metrics.loc[(metrics["Metric Name"] == "l2_read_transactions")]
    l2_read_stats = l2_read_stats.loc[(l2_read_stats.Kernel.str.match(r'{}.'.format(kernel_name), na=False))]
    l2_read_transactions = 0
    for tr in l2_read_stats["Avg"]:
        l2_read_transactions += int(tr)

    l2_write_stats = metrics.loc[(metrics["Metric Name"] == "l2_write_transactions")]
    l2_write_stats = l2_write_stats.loc[(l2_write_stats.Kernel.str.match(r'{}.'.format(kernel_name), na=False))]
    l2_write_transactions = 0
    for tr in l2_write_stats["Avg"]:
        l2_write_transactions += int(tr)

    l2_total = l2_read_transactions + l2_write_transactions 

    l2_intensity = kernel_dic[id]['total_inst'] / l2_total
    l2_performance = kernel_dic[id]['total_inst'] / (1000 * kernel_dic[id]['kernel_time']) # performance in  GIPS ( kernel_time in μsecs )

    graph.plot(l2_intensity, l2_performance, color='g', marker = 's', label="L2 (tot_inst)")

    ## HBM stats ##

    dram_read_stats = metrics.loc[(metrics["Metric Name"] == "dram_read_transactions")]
    dram_read_stats = dram_read_stats.loc[(dram_read_stats.Kernel.str.match(r'{}.'.format(kernel_name), na=False))]
    dram_read_transactions = 0
    for tr in dram_read_stats["Avg"]:
        dram_read_transactions += int(tr)

    dram_write_stats = metrics.loc[(metrics["Metric Name"] == "dram_write_transactions")]
    dram_write_stats = dram_write_stats.loc[(dram_write_stats.Kernel.str.match(r'{}.'.format(kernel_name), na=False))]
    dram_write_transactions = 0
    for tr in dram_write_stats["Avg"]:
        dram_write_transactions += int(tr)

    dram_total = dram_read_transactions + dram_write_transactions 

    hbm_intensity = kernel_dic[id]['total_inst'] / dram_total
    hbm_performance = kernel_dic[id]['total_inst'] / (1000 * kernel_dic[id]['kernel_time']) # performance in  GIPS ( kernel_time in μsecs )

    graph.plot(hbm_intensity, hbm_performance, color='b', marker = 's', label="HBM (tot_inst)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot roofline model')
    parser.add_argument('-kernel_name', type=str, help='Kernel name as shown in csv')
    parser.add_argument('-timings', nargs='+', help='Timing files (timing_kernel.csv)')
    parser.add_argument('-events', nargs='+', help='Event files (events_kernel.csv)')
    parser.add_argument('-metrics', nargs='+', help='Metric files (metrics_kernel.csv)')
    parser.add_argument('-title', type=str, help='Graph title')
    parser.add_argument('-out', type=str, help='Output file (default = roofline_kernel_name.png)')
    parser.add_argument('-simple', action='store_true', help='Plot only total transactions')
    args = parser.parse_args()

    timings = args.timings
    events = args.events
    metrics = args.metrics
    simple = args.simple
    
    kernel_name = args.kernel_name
    title = args.title
    out = args.out


    ax, fig = create_graph()
    
    kernel_dic = {}

    for file in timings:
        timing(kernel_dic, kernel_name, file)

    for file in events:
        find_inst(kernel_dic, kernel_name, file)

    for file in metrics:
        app_char(kernel_dic, kernel_name, file, ax, simple)
    
    # add title and legend
    ax.legend(loc='lower right')

    ax.set_title(title)
    ax.grid(True)

    figname = f"roofline_{kernel_name}.png" if out is None else out

    fig.savefig(figname, bbox_inches="tight")