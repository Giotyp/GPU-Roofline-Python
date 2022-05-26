import matplotlib
from matplotlib.transforms import Bbox
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def roofline(kernel_name):
    # Format csv files to discard nvprof unwanted output
    # e.g. : ==5328== NVPROF is profiling process 5328, command: ./transpose

    for file in [f'timing_{kernel_name}.csv', f'metrics_{kernel_name}.csv']:
        with open(file,"r+") as f:
            new_f = f.readlines()
            f.seek(0)
            for line in new_f:
                if "==" not in line:
                    f.write(line)
            f.truncate()

    # get csv databases with pandas
    timing = pd.read_csv(f'timing_{kernel_name}.csv')
    metrics = pd.read_csv(f'metrics_{kernel_name}.csv')

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
    unit = 'us'

    # Total Instructions

    instructions = metrics.loc[(metrics["Event Name"] == "thread_inst_executed")]

    # take Average if there are multiple kernel calls and scale to  warp-level
    total_inst_nrml = int(instructions["Avg"]) / 32 


    ## L1 stats ##

    gld_stats = metrics.loc[(metrics["Event Name"] == "gld_transactions")]
    gld_transactions = int(gld_stats["Avg"])

    gst_stats = metrics.loc[(metrics["Event Name"] == "gst_transactions")]
    gst_transactions = int(gst_stats["Avg"])

    sld_stats = metrics.loc[(metrics["Event Name"] == "shared_load_transactions")]
    sld_transactions = int(sld_stats["Avg"])

    sst_stats = metrics.loc[(metrics["Event Name"] == "shared_store_transactions")]
    sst_transactions = int(sst_stats["Avg"])

    l1_total = gld_transactions + gst_transactions + sld_transactions + sst_transactions

    l1_intensity = total_inst_nrml / l1_total
    l1_performance = total_inst_nrml / (1000 * kernel_time) # performance in  GIPS ( kernel_time in μsecs )


    ## Global Transactions ##

    gb_stats = metrics.loc[(metrics["Event Name"] == "inst_executed_global_loads")]
    gb_transactions = int(gb_stats["Avg"])

    gb_total = gld_transactions + gst_transactions
    gb_intensity = gb_transactions / gb_total
    gb_performance = gb_transactions / (1000 * kernel_time)


    ## L2 stats ##

    l2_read_stats = metrics.loc[(metrics["Event Name"] == "l2_read_transactions")]
    l2_read_transactions = int(l2_read_stats["Avg"])

    l2_write_stats = metrics.loc[(metrics["Event Name"] == "l2_write_transactions")]
    l2_write_transactions = int(l2_write_stats["Avg"])

    l2_total = l2_read_transactions + l2_write_transactions 

    l2_intensity = total_inst_nrml / l2_total
    l2_performance = total_inst_nrml / (1000 * kernel_time) # performance in  GIPS ( kernel_time in μsecs )


    ## HBM stats ##

    dram_read_stats = metrics.loc[(metrics["Event Name"] == "dram_read_transactions")]
    dram_read_transactions = int(dram_read_stats["Avg"])

    dram_write_stats = metrics.loc[(metrics["Event Name"] == "dram_write_transactions")]
    dram_write_transactions = int(dram_write_stats["Avg"])

    dram_total = dram_read_transactions + dram_write_transactions 

    hbm_intensity = total_inst_nrml / dram_total
    hbm_performance = total_inst_nrml / (1000 * kernel_time) # performance in  GIPS ( kernel_time in μsecs )


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

    # define ceilings
    help_x = np.asarray([instr_min, instr_max]) # helper dotted line
    help_y = np.asarray([l1_performance, l1_performance])

    peak_x = np.asarray([l1_elbow, 10**xmax]) # performance ceiling
    peak_y = np.asarray([peak, peak]) 

    l1_x = np.asarray([instr_min, l1_elbow ]) # L1 ceiling
    l1_y = np.asarray([l1_bw*instr_min, peak])

    l2_x = np.asarray([instr_min, l2_elbow]) # L2 ceiling
    l2_y = np.asarray([l2_bw*instr_min, peak])

    hbm_x = np.asarray([instr_min, hbm_elbow]) # HBM ceiling
    hbm_y = np.asarray([hbm_bw*instr_min, peak])

    # add architectural characterization to figure
    ax.plot(help_x, help_y, color='0', linestyle='--')
    ax.plot(peak_x, peak_y, color='0') # Performance ceiling
    ax.plot(l1_x, l1_y, color='r', label='L1 437.5 GTXN/s') # L1 ceiling
    ax.plot(l2_x, l2_y, color='g', label='L2 93.6 GTXN/s') # L2 ceiling
    ax.plot(hbm_x, hbm_y, color='b', label='HBM 25.9 GTXN/s') # HBM ceiling

    # add application characterization
    ax.plot(l1_intensity, l1_performance, color='r', marker = 's', label="L1 (tot_inst)")
    ax.plot(l2_intensity, l2_performance, color='g', marker = 's', label="L2 (tot_inst)")
    ax.plot(hbm_intensity, hbm_performance, color='b', marker = 's', label="HBM (tot_inst)")
    ax.plot(gb_intensity, gb_performance, color='y', marker = 's', label="Global (ldst_inst)")


    # add title, optional text for peak and labels
    ax.text(l1_elbow, peak+100, 'Theoretical Peak: 489.6 warp GIPS')
    ax.legend(loc='lower right')

    ax.set_title("Matrix Transpose Naive: global memory pattern")
    ax.grid(True)

    figname = f"roofline_{kernel_name}.png"
    fig.savefig(figname, bbox_inches="tight")


if __name__ == "__main__":
    roofline('transpose') # define kernel name used in csv files  