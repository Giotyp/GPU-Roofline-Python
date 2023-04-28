import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
}


# Function to return float value from row object
# ncu stores numbers in ',' format e.g. 750,000.0
def float_val(x):
    return float(x.to_string().split(' ')[-1].replace(',',''))


def create_graph(memories):
    ## Define Bandwidths ##

    #peak = 489.6 # theoretical peak for V100S in warp GIPS
    peak = 609.12 # theoretical peak for A100

    #l1_bw = 437.5 # theoretical bandwidth of L1 for V100S
    l1_bw = 1312.5 # theoretical bandwidth of L1 for A100
    l1_elbow = peak/l1_bw

    #l2_bw = 93.6 # theoretical bandwidth of L2 for V100S
    l2_bw = 215.3 # theoretical bandwidth of L2 for A100
    l2_elbow = peak/l2_bw

    #hbm_bw = 25.9 # theoretical bandwidth of HBM for V100S
    hbm_bw = 48.6  # theoretical bandwidth of HBM for A100
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

    l1, l2, hbm = memories

    ax.plot(peak_x, peak_y, color='0') # Performance ceiling

    if l1:
        ax.plot(l1_x, l1_y, color='r', label=f'L1 {l1_bw} GTXN/s') # L1 ceiling
    if l2:
        ax.plot(l2_x, l2_y, color='g', label=f'L2 {l2_bw} GTXN/s') # L2 ceiling        
    if hbm:    
        ax.plot(hbm_x, hbm_y, color='b', label=f'HBM {hbm_bw} GTXN/s') # HBM ceiling

    # text for peak performance
    ax.text(l1_elbow, peak+100, f'Theoretical Peak: {peak} warp GIPS')

    return ax,fig



def timing(kernel_stats, kernel_name, time_file, profiler):
    # Format csv files to discard nvprof unwanted output
    # e.g. : ==5328== NVPROF is profiling process 5328, command: ./transpose

    with open(time_file,"r+") as f:
        new_f = f.readlines()
        f.seek(0)
        for line in new_f:
            if "==" not in line:
                f.write(line)
        f.truncate()

    # get csv database with pandas
    timing = pd.read_csv(time_file)

    ## Kernel Time ##
    time_row = timing.loc[(timing[profiler["time_kernel"]] == kernel_name)]

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
    with open(events_file,"r+") as f:
        new_f = f.readlines()
        f.seek(0)
        for line in new_f:
            if "==" not in line:
                f.write(line)
        f.truncate()

    events = pd.read_csv(events_file)

    # Total Instructions
    instructions = events.loc[(events[profiler["metric_kernel"]] == kernel_name)]
    kernel_inst = instructions[profiler["Average"]]
    kernel_inst = float_val(kernel_inst)
    total_inst_nrml = kernel_inst

    total_inst_nrml /= 32

    kernel_stats['total_inst'] = total_inst_nrml



def app_char(kernel_stats, kernel_name, metrics_file, graph, memories, profiler, labels, colors, markers, mode):
    with open(metrics_file,"r+") as f:
        new_f = f.readlines()
        f.seek(0)
        for line in new_f:
            if "==" not in line:
                f.write(line)
        f.truncate()

    metrics = pd.read_csv(metrics_file)
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


if __name__ == "__main__":

    ## Define dictionaries for colors, markers, labels ##
    
    colors = {
        "l1":"r",
        "l2":"g",
        "hbm":"b",
        'kernel_A': 'y',
        "kernel_B":'m',
    }

    markers = {
        "l1":"s",
        "l2":"s",
        "hbm":"s",
        'kernel_A': 's',
        "kernel_B":'s',
    }

    labels = {
        "l1":"L1 (tot_inst)",
        "l2":"L2 (tot_inst)",
        "hbm":"HBM (tot_inst)",
        'kernel_A': "A_kernel",
        'kernel_B': "B_kernel",
    }

    ## Define csv filenames ##

    timings = "multi_kernels/timing.csv"
    events = "multi_kernels/events.csv"
    metrics = "multi_kernels/metrics.csv"

    ## Define profiler (ncu or nvp)     ##
    ## Dictionaries declared at the top ##

    profiler = ncu

    ## Define kernel name(s) (as shown in csv file)     ##
    ## For hierarchical roofline provide single kernel  ##
    ## otherwise provide multiple kernels               ##     

    kernels = ["kernel_A", "kernel_B"]

    ## Define Memories for ceilings ##

    l1, l2, hbm = True, True, True
    memories_ceil = [l1, l2, hbm]

    ## Define Memories for plotting ##

    l1, l2, hbm = True, False, False
    memories_plot = [l1, l2, hbm]

    ## Define graph elements ##

    title = f"Kernel L1 Performance (NVIDIA A100)"
    figname = f"multi_kernels/roofline_kernels.png"

    ## Define rooflinemode           ##
    ## (0: hierarchical for 1 kernel ##
    ##  1: multiple kernels)         ##

    mode = 1
    

    ax, fig = create_graph(memories_ceil)
    

    for kernel_name in kernels:
        kernel_stats = {}

        timing(kernel_stats, kernel_name, timings, profiler)

        find_inst(kernel_stats, kernel_name, events, profiler)

        app_char(kernel_stats, kernel_name, metrics, ax, memories_plot, profiler, labels, colors, markers, mode)
        
        # add legend
        ax.legend(loc='lower right', fontsize='10')


    ax.set_title(title)
    ax.grid(True)

    fig.savefig(figname, bbox_inches="tight")