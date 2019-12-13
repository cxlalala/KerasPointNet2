TF_FLAGS := $(shell python -c 'import tensorflow as tf;print(" ".join(tf.sysconfig.get_compile_flags()) + " " + " ".join(tf.sysconfig.get_link_flags()))')
CUDA_LINK := -lcudart -L /usr/local/cuda-10.0/lib64/
CUDA_INCLUDE := -I /usr/local/cuda-10.0/include
CUDA_NVCC := /usr/local/cuda-10.0/bin/nvcc
DEFAULT_CPP_FLAGS := -std=c++11 -shared -fPIC $(CUDA_LINK) $(CUDA_INCLUDE) $(TF_FLAGS) -O2 -D_GLIBCXX_USE_CXX11_ABI=0
DEFAULT_NVCC_FLAGS := -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

grouping_objects := grouping_so.so grouping.cu.o
interpolation_objects := interpolation_so.so interpolation.cu.o
sampling_objects := sampling_so.so sampling.cu.o
objects = $(grouping_objects) $(interpolation_objects) $(sampling_objects)

.PHONY: all
all: $(objects)

$(grouping_objects):
	nvcc grouping.cu -o grouping.cu.o $(DEFAULT_NVCC_FLAGS)
	c++ grouping.cpp grouping.cu.o -o grouping_so.so $(DEFAULT_CPP_FLAGS)

$(sampling_objects):
	nvcc sampling.cu -o sampling.cu.o $(DEFAULT_NVCC_FLAGS)
	c++ sampling.cpp sampling.cu.o -o sampling_so.so $(DEFAULT_CPP_FLAGS)

$(interpolation_objects):
	nvcc interpolation.cu -o interpolation.cu.o $(DEFAULT_NVCC_FLAGS)
	c++ interpolation.cpp interpolation.cu.o -o interpolation_so.so $(DEFAULT_CPP_FLAGS)

.PHONY: clean
clean:
	rm -f $(objects)
