GENCODE_SM30 := -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 := -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50 := -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM61 := -gencode=arch=compute_61,code=\"sm_61,compute_61\"
GENCODE_SM70 := -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM80 := -gencode=arch=compute_80,code=\"sm_80,compute_80\"
GENCODE_SM89 := -gencode=arch=compute_89,code=\"sm_89,compute_89\"
GENCODE_SM90 := -gencode=arch=compute_90,code=\"sm_90,compute_90\"
# Blackwell (requires CUDA 12.8+); uncomment if your toolkit supports it
#GENCODE_SM100 := -gencode=arch=compute_100,code=\"sm_100,compute_100\"
#GENCODE_SM120 := -gencode=arch=compute_120,code=\"sm_120,compute_120\"

CUOPTS = $(GENCODE_SM70) $(GENCODE_SM80) $(GENCODE_SM89) $(GENCODE_SM90)


CC := nvcc

INCLUDE := 
LIB := 

SRC = l1_bw_64f.cu

EXE = l1_bw_64f

release:
	$(CC) --ptxas-options=-v $(CUOPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) -lcudart

clean:
	rm -f *.o; rm -f $(EXE)

run:
	./$(EXE)

profile:
	nvprof ./$(EXE)

events:
	nvprof  --events elapsed_cycles_sm ./$(EXE)
