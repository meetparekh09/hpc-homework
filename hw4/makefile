CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

# turning off auto-vectorization since this can make hand-vectorized code slower
CXXFLAGS += -fno-tree-vectorize

NVCC = nvcc
NVCCFLAGS = -std=c++11
NVCCFLAGS += -Xcompiler "-fopenmp" # pass -fopenmp to host compiler (g++)

TARGETS =  $(basename $(wildcard *.cpp)) $(basename $(wildcard *.c)) $(basename $(wildcard *.cu))

all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

%:%.c
	$(CXX) $(CXXFLAGS) $(CUDA_INCDIR) $< $(CUDA_LIBS) -o $@

%:%.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
