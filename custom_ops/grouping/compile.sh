#!/usr/bin/env bash
$CUDA_NVCC grouping.cu -o grouping.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
$CXX -std=c++11 grouping.cpp grouping.cu.o -o grouping_so.so -shared -fPIC $CUDA_LINK $CUDA_INCLUDE ${TF_LFLAGS[@]} ${TF_CFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
