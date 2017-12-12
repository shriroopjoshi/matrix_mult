NVCC=nvcc
CUDAFLAGS= -arch=compute_35 -code=sm_35

kernel: kernel.cu
	$(NVCC) $(CUDAFLAGS) $^ -o $@

all: kernel

clean: 
	rm -f kernel