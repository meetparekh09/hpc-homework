#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int N = 100;
    int rank, size;
    int sum_static;
    int sum;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next = (rank+1)%size;
    if(rank == 0) {
        sum_static = N*size*(size-1)/2;
        sum = 0;
    }

    for(int i = 0; i < N; i++) {
        if(rank == 0) {
            sum += rank;
            MPI_Send(&sum, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
            MPI_Recv(&sum, 1, MPI_INT, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("Process %d received Integer %d from %d\n", rank, data, size-1);
        } else {
            MPI_Recv(&sum, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("Process %d received Integer %d from %d\n", rank, data, rank-1);
            sum += rank;
            MPI_Send(&sum, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        }
    }

    if(rank == 0) {
        printf("Static Sum :: %d\n", sum_static);
        printf("Parallel Sum :: %d\n", sum);
        printf("Error :: %d\n", abs(sum_static - sum));
    }

    MPI_Finalize();
    return 0;
}
