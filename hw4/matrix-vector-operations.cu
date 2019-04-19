#include <stdio.h>
#include <omp.h>
#include <algorithm>


#define BLOCK_SIZE 1024
#define MULTIDIM_BLOCK_SIZE 32

void inner_product(double *x, double *y, double *ans_ptr, long N) {
    double ans = 0.0;
    #pragma omp parallel for reduction(+:ans)
    for(long i = 0; i < N; i++) {
        ans += x[i]*y[i];
    }
    *ans_ptr = ans;
}

void matrix_vector_mult(double *x, double *y, double *ans_ptr, long M, long N) {
    #pragma omp parallel for
    for(int i = 0; i < M; i++) {
        double ans = 0.0;
        #pragma omp parallel for reduction(+:ans)
        for(int j = 0; j < N; j++) {
            ans += x[N*i + j] * y[j];
        }
        ans_ptr[i] = ans;
    }
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

__global__ void matrix_vector_mult_kernel(double *a, double *b, double *c, long M, long N) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("%d %d\n", x_idx, y_idx);

    if(x_idx < N && y_idx < M) c[y_idx * N + x_idx] = a[y_idx * N + x_idx] * b[x_idx];
}

void execute_dot_product() {
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

}

void execute_matrix_vector_multiplication() {
    long M = (1UL<<13);
    long N = (1UL<<13);

    double *x, *y, *ans_ref;

    cudaMallocHost((void**)&x, M * N * sizeof(double));
    cudaMallocHost((void**)&y, N * sizeof(double));
    cudaMallocHost((void**)&ans_ref, M * sizeof(double));

    #pragma omp parallel for
    for(int j = 0; j < N; j++) {
        for(long i = 0; i < M; i++) {
            x[N*i + j] = i + j;
        }
        y[j] = j;
    }

    double tt = omp_get_wtime();
    matrix_vector_mult(x, y, ans_ref, M, N);
    double te = omp_get_wtime();

    printf("CPU Performance = %f GFLOPs\n", 2*M*N*sizeof(double) / 1e9 / (te-tt));
    printf("CPU Bandwidth = %f GB/s\n", 1*M*N*sizeof(double) / (te-tt)/1e9);


    double *a, *b, *c, *c_h, *sum, *sum_d;
    cudaMalloc(&a, M * N * sizeof(double));
    cudaMalloc(&b, N * sizeof(double));
    cudaMalloc(&c, M * N * sizeof(double));
    cudaMallocHost((void**)&c_h, M * sizeof(double));

    long N_work = 1;
    for (long j = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); j > 1; j = (j+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += j;
    cudaMalloc(&sum, N_work*sizeof(double));

    cudaMemcpyAsync(a, x, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(b, y, N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimGrid((M + MULTIDIM_BLOCK_SIZE - 1) / MULTIDIM_BLOCK_SIZE, (N + MULTIDIM_BLOCK_SIZE - 1) / MULTIDIM_BLOCK_SIZE);
    dim3 dimBlock(MULTIDIM_BLOCK_SIZE, MULTIDIM_BLOCK_SIZE);


    tt = omp_get_wtime();

    matrix_vector_mult_kernel<<<dimGrid, dimBlock>>>(a, b, c, M, N);

    for(int i = 0; i < M; i++) {
        long N_temp = N;

        sum_d = sum;
        long Nb = (N_temp + BLOCK_SIZE - 1)/BLOCK_SIZE;
        reduction_kernel2<<<Nb, BLOCK_SIZE>>>(sum_d, c + i*N, N_temp);

        while(Nb > 1) {
            N_temp = Nb;
            Nb = (Nb + BLOCK_SIZE - 1)/BLOCK_SIZE;
            reduction_kernel2<<<Nb, BLOCK_SIZE>>>(sum_d + N_temp, sum_d, N_temp);
            sum_d += N_temp;
        }

        cudaDeviceSynchronize();
        cudaMemcpyAsync(c_h + i, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    te = omp_get_wtime();
    printf("GPU Performance = %f GFLOPs\n", 2*M*N*sizeof(double) / 1e9 / (te-tt));
    printf("GPU Bandwidth = %f GB/s\n", 1*M*N*sizeof(double) / (te - tt)/1e9);

    double error = 0.0;
    for(int i = 0; i < M; i++) {
        error += fabs(c_h[i] - ans_ref[i]);
    }

    printf("Error: %f\n", error);
}

int main() {
    printf("Taking Inner Product :: \n");
    execute_dot_product();

    printf("\n\n ==================================================================================== \n\n");

    printf("Taking Matrix Vector Product :: \n");
    execute_matrix_vector_multiplication();
}
