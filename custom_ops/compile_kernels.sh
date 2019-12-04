TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
CUDA_LINK="-lcudart -L /usr/local/cuda-10.0/lib64/"
CUDA_INCLUDE="-I /usr/local/cuda-10.0/include"
CUDA_NVCC="/usr/local/cuda-10.0/bin/nvcc"
# Note: If using tensorflow 1.14, you MUST use g++ 4.8 otherwise it segfaults.
#CXX="g++-4.8"
CXX="g++"

while read -r dir; do
    pushd $dir
    source ./compile.sh &
    popd
done << EOF
./sampling/
./grouping
./interpolation/
EOF
