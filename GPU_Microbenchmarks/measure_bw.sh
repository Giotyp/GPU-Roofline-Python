#!/bin/bash

# Use GPU_Microbenchmarks from github.com/accel-sim/gpu-app-collection/tree/release/src/cuda/GPU_Microbenchmark
# to measure the memory bandwidth of L1, L2 cache and HBM. Also measure Max GPU IPS.
# To calculate GPU frequency we use nvidia-smi

# Make sure to provide all Makefiles with the appropriate GENCODE_SMxx where xx is GPU compute capability 

L1_path='l1_bw_64f'
L1_bin='l1_bw_64f'

L2_path='l2_bw_64f'
L2_bin='l2_bw_64f'

HBM_path='mem_bw'
HBM_bin='mem_bw'

Flops_path='MaxFlops'
Flops_bin='MaxFlops'

# Execute Makefiles
make -s -C $L1_path
make -s -C $L2_path
make -s -C $HBM_path
make -s -C $Flops_path

echo -e "Running GPU Benchmarks \n" > bw_measurements.txt
 
# Calculate bandwidths
./$L1_path/$L1_bin >> bw_measurements.txt
echo "" >> bw_measurements.txt # leave blank line
./$L2_path/$L2_bin >> bw_measurements.txt
echo "" >> bw_measurements.txt
./$HBM_path/$HBM_bin >> bw_measurements.txt
echo "" >> bw_measurements.txt
./$Flops_path/$Flops_bin >> bw_measurements.txt

# L1 BW
L1_bw=$(grep -E 'L1 bandwidth' bw_measurements.txt | cut -d '=' -f 2 | cut -d '(' -f 1)
## multiply for n SM's . V100 has 80 SM 
SM=80
L1_bw=$(echo ${L1_bw} \* ${SM} | bc)

# L2 BW
L2_bw=$(grep -E 'L2 bandwidth' bw_measurements.txt | cut -d '=' -f 2 | cut -d '(' -f 1)

# HBM BW
HBM_bw=$(grep -E 'GB/sec' bw_measurements.txt | cut -d '=' -f 2 | cut -d '(' -f 1)

# Max Flops
MaxFlops=$(grep -E 'FLOP per SM' bw_measurements.txt | cut -d '=' -f 2 | cut -d '(' -f 1)
MaxFlops=$(echo ${MaxFlops} \* ${SM} | bc)

# Obtain GPU SM Max frequency
frequency=$(nvidia-smi -q -d CLOCK | sed -n '/SM Clock Samples/, /Avg/p' | grep Max | cut -d ':' -f 2 | cut -d 'M' -f 1)

echo -e "\nFinal Measurements After Calculations\n" >> bw_measurements.txt

# Multiply L1,L2 BW with frequency -> GB/sec
frequency=$(echo ${frequency} \* 0.001 | bc) # transform to GHz
L1_bw=$(echo ${L1_bw} \* ${frequency} | bc)
L2_bw=$(echo ${L2_bw} \* ${frequency} | bc)
MaxFlops=$(echo ${MaxFlops} \* ${frequency} | bc)

echo "L1 bandwidth = ${L1_bw} (GB/sec)" >> bw_measurements.txt  
echo "L2 bandwidth = ${L2_bw} (GB/sec)" >> bw_measurements.txt
echo "HBM bandwidth = ${HBM_bw} (GB/sec)" >> bw_measurements.txt
echo "Max Flops = ${MaxFlops} (GFLOPS)" >> bw_measurements.txt

echo -e "\nMeasurements normalized to 32B transaction size\n" >> bw_measurements.txt

# multiply with 1/32 = 0.03125
L1_bw=$(echo ${L1_bw} \* 0.03125 | bc)
L2_bw=$(echo ${L2_bw} \* 0.03125 | bc)
HBM_bw=$(echo ${HBM_bw} \* 0.03125 | bc)
MaxFlops=$(echo ${MaxFlops} \* 0.03125 | bc)

echo "L1 bandwidth = ${L1_bw} (GTXN/s)" >> bw_measurements.txt  
echo "L2 bandwidth = ${L2_bw} (GTXN/s)" >> bw_measurements.txt
echo "HBM bandwidth = ${HBM_bw} (GTXN/s)" >> bw_measurements.txt
echo "Max warp-based Instructions = ${MaxFlops} (GIPS)" >> bw_measurements.txt