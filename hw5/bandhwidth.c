#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int N = 10000;
    int n = 2000000/sizeof(int);
    int rank, size;
    int data[n];
    int recv[n];
    clock_t ts, te;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for(int i = 0; i < n; i++) {
        data[i] = i;
    }

    int next = (rank+1)%size;
    if(rank == 0) {
        ts = clock();
    }

    for(int i = 0; i < N; i++) {
        if(rank == 0) {
            MPI_Send(&data, n, MPI_INT, next, 0, MPI_COMM_WORLD);
            MPI_Recv(&recv, n, MPI_INT, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(&recv, n, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&data, n, MPI_INT, next, 0, MPI_COMM_WORLD);
        }
    }

    if(rank == 0) {
        te = clock();
        double time = (double)(te - ts)/CLOCKS_PER_SEC;
        double bandwidth = 2e6 * size * N / time / 1e9;
        printf("N :: %d, Latency :: %f\n", N, time);
        printf("Bandwidth :: %f GB/s\n", bandwidth);
    }

    MPI_Finalize();
    return 0;
}
