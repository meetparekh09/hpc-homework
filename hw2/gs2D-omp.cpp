#include <stdio.h>
#include <math.h>
#include "utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define LIMIT 10000
#define TOLERANCE 1e-8

// double h = 1.0;
// long n = 0;


int main(int argc, char **argv) {
    long n = read_option<long>("-N", argc, argv);
    double h = 1.0/(n+1);

    double *u = (double*) malloc(n * n * sizeof(double));
    double *t = (double*) malloc(n * n * sizeof(double));

    #ifdef _OPENMP
    printf("Threads in use :: %d\n", omp_get_num_threads());
    #pragma omp parallel for
    #endif
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            u[i + n*j] = 0;
            t[i + n*j] = 0;
        }
    }

    Timer timer;

    timer.tic();
    int iter = 0;
    double error = 1e9;
    while((iter < LIMIT) && (error > TOLERANCE)) {

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(int i = 1; i < n-1; i++) {
            for(int j = 1; j < n-1; j++) {
                if((i+j)%2 == 0)
                    t[i + n*j] = (h*h + u[i-1 + n*j] + u[i + n*(j-1)] + u[i+1 + n*j] + u[i + n*(j-1)])/4;
            }
        }

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(int i = 1; i < n-1; i++) {
            for(int j = 1; j < n-1; j++) {
                if((i+j)%2 == 1)
                    t[i + n*j] = (h*h + t[i-1 + n*j] + t[i + n*(j-1)] + t[i+1 + n*j] + t[i + n*(j-1)])/4;
            }
        }

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                u[i + n*j] = t[i + n*j];
            }
        }

        if(iter % 100 == 0) {
            printf("Iter :: %d\n", iter);
        }


        iter += 1;

    }

    printf("Time for %d iterations, for n : %ld is %f", iter, n, timer.toc());
    return 0;
}
