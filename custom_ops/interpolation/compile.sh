#!/usr/bin/env bash
$CXX -std=c++11 interpolation.cpp -o interpolation_so.so -shared -fPIC $CUDA_LINK $CUDA_INCLUDE ${TF_LFLAGS[@]} ${TF_CFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0