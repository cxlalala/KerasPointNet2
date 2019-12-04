// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void threenn_gpu(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
    int batch_index = blockIdx.x;
    int index = threadIdx.x;
    int stride = blockDim.x;
}

void threennGPULauncher(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
    threenn_gpu<<<b,256>>>(b,n,m,xyz1,xyz2,dist,idx);
}
