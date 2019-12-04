#!/usr/bin/env bash
$CUDA_NVCC interpolation.cu -o interpolation.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
$CXX -std=c++11 interpolation.cpp interpolation.cu.o -o interpolation_so.so -shared -fPIC $CUDA_LINK $CUDA_INCLUDE ${TF_LFLAGS[@]} ${TF_CFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
