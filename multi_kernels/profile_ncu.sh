#!/bin/bash

# Draft script profiling a python application 
# that utilizes a GPU model (e.g. with CuPy library)
# the Nsight Compute Profiler (ncu) is used
# exe file must be defined appropriately


gld_transactions='l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'
gst_transactions='l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum'
shared_load_transactions='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'
shared_store_transactions='l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum'
l2_transactions_2='lts__t_sectors_op_atom.sum'
l2_transactions_3='lts__t_sectors_op_red.sum'
l2_read_transactions_1='lts__t_sectors_op_read.sum'
l2_read_transactions="${l2_read_transactions_1},${l2_transactions_2},${l2_transactions_3}"
l2_write_transactions_1='lts__t_sectors_op_write.sum'
l2_write_transactions="${l2_write_transactions_1},${l2_transactions_2},${l2_transactions_3}"
dram_read_transactions='dram__sectors_read.sum'
dram_write_transactions='dram__sectors_write.sum'

metrics="${gld_transactions},${gst_transactions},${shared_load_transactions},${shared_store_transactions},${l2_read_transactions},${l2_write_transactions},${dram_read_transactions},${dram_write_transactions}"
events='smsp__thread_inst_executed.sum'

options='--csv --profile-from-start off --print-summary per-kernel'

$ncu $options --metrics gpu__time_active.avg --log-file timing.csv  $python $exe
$ncu $options --metrics $metrics --log-file metrics.csv $python $exe
$ncu $options --metrics $events   --log-file events.csv  $python $exe