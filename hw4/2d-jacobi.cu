#include <stdio.h>
#include <omp.h>

#define ITERS 100
#define BLOCK_SIZE 1024

__global__ void jacobi_kernel(double *u,double *result, double h, int N){
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double f_h;
    f_h = h*h;

    if(idx <= N) {
        return;
    }
    if(idx % N == 0) {
        return;
    }
    if(idx % N == (N-1)) {
        return;
    }
    if(idx >= N*N - N) {
        return;
    }

    result[idx] = (f_h + u[idx - N] + u[idx - 1] + u[idx + N] + u[idx + 1])/4;
}

int main() {
    long N = (1UL<<13);
    double h = 1.0/(N+1);

    double *u_odd, *u_eve;
    cudaMallocHost((void**)&u_odd, N * N * sizeof(double));
    cudaMallocHost((void**)&u_eve, N * N * sizeof(double));

    #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if(i == 0 || j == 0 || i == N-1 || j == N-1) {
                u_odd[N*i + j] = u_eve[N*i + j] = 0;
            } else {
                u_odd[N*i + j] = u_eve[N*i + j] = 1;
            }
        }
    }

    double *d_u_odd, *d_u_eve;
    cudaMalloc(&d_u_odd, N * N * sizeof(double));
    cudaMalloc(&d_u_eve, N * N * sizeof(double));
    cudaMemcpyAsync(d_u_odd, u_odd, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_u_eve, u_eve, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    int i = 0;

    double tt = omp_get_wtime();

    while(i < ITERS) {
        if(i%2 == 0) {
            jacobi_kernel<<<(N*N + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_u_odd, d_u_eve, h, N);
        } else {
            jacobi_kernel<<<(N*N + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_u_eve, d_u_odd, h, N);
        }
        i++;
    }

    double te = omp_get_wtime();

    printf("Runtime :: %f\n", te - tt);
    printf("Number of Iterations :: %d\n", ITERS);
    printf("Dimensions :: %d x %d\n", N, N);

}
