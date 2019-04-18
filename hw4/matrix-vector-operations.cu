#include <stdio.h>
#include <omp.h>
#include <algorithm>


#define BLOCK_SIZE 1024

void inner_product(double *x, double *y, double *ans_ptr, long N) {
    double ans = 0.0;
    #pragma omp parallel for reduction(+:ans)
    for(long i = 0; i < N; i++) {
        ans += x[i]*y[i];
    }
    *ans_ptr = ans;
}


__global__ void reduction_kernel2(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void inner_product_multiply_kernel(double *a, double *b, double *c, long N) {
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

    if(idx < N) c[idx] = a[idx] * b[idx];
}


int main() {
    long N = (1UL<<25);
    double *x, *y;

    cudaMallocHost((void**)&x, N * sizeof(double));
    cudaMallocHost((void**)&y, N * sizeof(double));

    #pragma omp parallel for
    for(long i = 0; i < N; i++) {
        x[i] = 1.0/(i+1);
        y[i] = 3.0/(i+2);
    }

    double ans_ref, ans;
    double tt = omp_get_wtime();
    inner_product(x, y, &ans_ref, N);
    double te = omp_get_wtime();

    printf("CPU Performance = %f GFLOPs\n", 2*N*sizeof(double) / 1e9 / (te-tt));
    printf("CPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (te-tt)/1e9);


    double *a, *b, *c, *sum, *sum_d;
    cudaMalloc(&a, N*sizeof(double));
    cudaMalloc(&b, N*sizeof(double));
    cudaMalloc(&c, N*sizeof(double));

    cudaMemcpyAsync(a, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(b, y, N*sizeof(double), cudaMemcpyHostToDevice);

    long N_work = 1;
    for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
    cudaMalloc(&sum, N_work*sizeof(double));

    cudaDeviceSynchronize();

    tt = omp_get_wtime();

    inner_product_multiply_kernel<<<(N + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(a, b, c, N);

    sum_d = sum;
    long Nb = (N + BLOCK_SIZE - 1)/BLOCK_SIZE;
    reduction_kernel2<<<Nb, BLOCK_SIZE>>>(sum_d, c, N);

    while(Nb > 1) {
        N = Nb;
        Nb = (Nb + BLOCK_SIZE - 1)/BLOCK_SIZE;
        reduction_kernel2<<<Nb, BLOCK_SIZE>>>(sum_d + N, sum_d, N);
        sum_d += N;
    }

    cudaMemcpyAsync(&ans, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    te = omp_get_wtime();

    N = (1UL<<25);

    printf("GPU Performance = %f GFLOPs\n", 2*N*sizeof(double) / 1e9 / (te-tt));
    printf("GPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (te - tt)/1e9);
    printf("Error = %f\n", fabs(ans-ans_ref));

    //printf("%ld\n", N);
}
