#!/bin/bash

cd example # modify appropriately 
kernel_name="transpose" # modify appropriately

metrics='inst_executed_global_loads,gld_transactions,gst_transactions,shared_load_transactions,shared_store_transactions,l2_read_transactions,l2_write_transactions,dram_read_transactions,dram_write_transactions'

events='thread_inst_executed'
# if profiling python module it may need the following option
# options='--openacc-profiling off'

nvprof --csv --print-gpu-summary --log-file timing_${kernel_name}.csv  ./${kernel_name}
nvprof  --csv --metrics $metrics --log-file metrics_${kernel_name}.csv ./${kernel_name}
nvprof  --csv --events $events   --log-file events_${kernel_name}.csv ./${kernel_name}

python3 ../plot.py -kernel_name transpose -events events_transpose.csv -timings timing_transpose.csv \
-metrics metrics_transpose.csv -title="Matrix Transpose Naive"
