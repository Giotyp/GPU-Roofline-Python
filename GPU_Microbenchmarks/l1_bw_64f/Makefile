GENCODE_SM30 := -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 := -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50 := -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM61 := -gencode=arch=compute_61,code=\"sm_61,compute_61\"
GENCODE_SM70 := -gencode=arch=compute_70,code=\"sm_70,compute_70\"

CUOPTS = $(GENCODE_SM35)  $(GENCODE_SM50) $(GENCODE_SM61) $(GENCODE_SM70) 


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
