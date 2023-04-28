#!/bin/bash

# Script profiling a CUDA application (transpose) 
# A python application can be profiled likewise
# the nvprof profiler is used

exe="transpose" # modify appropriately

# metrics for nvprof
metrics='inst_executed_global_loads,gld_transactions,gst_transactions,shared_load_transactions,shared_store_transactions,l2_read_transactions,l2_write_transactions,dram_read_transactions,dram_write_transactions'

events='thread_inst_executed'
# if profiling python module it may need the following option
# options='--openacc-profiling off'

nvprof --csv --print-gpu-summary --log-file timing_${exe}.csv  ./${exe}
nvprof  --csv --metrics $metrics --log-file metrics_${exe}.csv ./${exe}
nvprof  --csv --events $events   --log-file events_${exe}.csv ./${exe}
