#!/bin/bash

cd example # modify appropriately 
kernel_name="transpose" # modify appropriately

metrics='inst_executed_global_loads,gld_transactions,gst_transactions,shared_load_transactions,shared_store_transactions,l2_read_transactions,l2_write_transactions,dram_read_transactions,dram_write_transactions'

events='thread_inst_executed'

nvprof --csv --print-gpu-summary --log-file timing_${kernel_name}.csv ./${kernel_name}
nvprof  --csv --metrics $metrics --events $events --log-file metrics_${kernel_name}.csv ./${kernel_name}

python3 ../plot.py
