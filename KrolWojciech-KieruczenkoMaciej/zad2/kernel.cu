__global__ void
decompositionKernel(double *deviceMtrx, int mtrxSize, int idx, int maxIndex)
{
    //Get thread id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //Calculate first an last index for thread
    while (tid < maxIndex)
    {
        int startIdx = ((idx + tid + 1) * mtrxSize + idx);
        int endIdx = ((idx + tid + 1) * mtrxSize + mtrxSize);
        for (int i = startIdx + 1; i < endIdx; i++)
        {
            //Perform next decomposition operation
            deviceMtrx[i] = deviceMtrx[i]-(deviceMtrx[startIdx] * deviceMtrx[(idx * mtrxSize)+(idx + (i - startIdx))]);
        }
        tid += blockDim.x * gridDim.x; 
    }
}