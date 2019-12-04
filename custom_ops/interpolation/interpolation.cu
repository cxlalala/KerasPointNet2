// input: xyz1 (b,n,3), xyz2(b,m,3)
// output: dist (b,n,3), idx (b,n,3)
__global__ void threenn_gpu(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    idx += n*3*batch_index;
    dist += n*3*batch_index;

    xyz2 += m*3*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index; j<n; j+=stride) {
        float x1=xyz1[j*3+0];
        float y1=xyz1[j*3+1];
        float z1=xyz1[j*3+2];
        double best1=1e40; double best2=1e40; double best3=1e40;
        int besti1=0; int besti2=0; int besti3=0;
        for (int k=0;k<m;++k) {
            float x2=xyz2[k*3+0];
            float y2=xyz2[k*3+1];
            float z2=xyz2[k*3+2];
            //float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
            double d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            if (d<best1) {
                best3=best2;
                besti3=besti2;
                best2=best1;
                besti2=besti1;
                best1=d;
                besti1=k;
            } else if (d<best2) {
                best3=best2;
                besti3=besti2;
                best2=d;
                besti2=k;
            } else if (d<best3) {
                best3=d;
                besti3=k;
            }
        } 
        dist[j*3]=best1;
        idx[j*3]=besti1;
        dist[j*3+1]=best2;
        idx[j*3+1]=besti2;
        dist[j*3+2]=best3;
        idx[j*3+2]=besti3;
    } 
}

void threennGPULauncher(int b, int n, int m, const float *xyz1, const float *xyz2, float *dist, int *idx) {
    threenn_gpu<<<b,512>>>(b,n,m,xyz1,xyz2,dist,idx);
}
