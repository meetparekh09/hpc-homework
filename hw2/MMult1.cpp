// g++ -fopenmp -O3 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 48

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  // TODO: See instructions below

  // Best performance is for present combination.
  // Reason: variables 'j' and 'p' switches the columns
  // and since we are storing everything in column major order, this would mean lots of cache misses.
  // But 'i' does not change columns and thus it results in less misses,
  // also 'p' changes column of only 'a', while 'j' changes column of both 'b' and 'c'
  // therefore having 'j' after 'p' would result in more misses thus the best combination is 'j' followed by 'p' followed by 'i'.
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        c[i+j*m] = c[i+j*m] + b[p+j*k] * a[i+p*m];
      }
    }
  }
}


//Helper Function for MMult2
void read_write(int row_offset, int col_offset, long m, double *block, double *matrix, bool is_read) {
    for(int i = 0; i < BLOCK_SIZE; i++) {
        for(int j = 0; j < BLOCK_SIZE; j++) {
            int row = row_offset * BLOCK_SIZE + j;
            int col = col_offset * BLOCK_SIZE + i;
            int idx1 = i*BLOCK_SIZE + j;
            int idx2 = col*m + row;
            if(is_read)
                block[idx1] = matrix[idx2];
            else
                matrix[idx2] = block[idx1];
        }
    }
}

void MMult2(long m, long n, long k, double *a, double *b, double *c) {
    double *a_block = (double*)aligned_malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
    double *b_block = (double*)aligned_malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
    double *c_block = (double*)aligned_malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
    long block_num = m/BLOCK_SIZE;

    for(int j = 0; j < block_num; j++) {
        for(int i = 0; i < block_num; i++) {
            read_write(i, j, m, c_block, c, true);
            for(int p = 0; p < block_num; p++) {
                read_write(i, p, m, a_block, a, true);
                read_write(p, j, m, b_block, b, true);
                MMult1(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, a_block, b_block, c_block);
                read_write(i, j, m, c_block, c, false);
            }
        }
    }

    aligned_free(a_block);
    aligned_free(b_block);
    aligned_free(c_block);
}

void MMult3(long m, long n, long k, double *a, double *b, double *c) {
    double *a_block = (double*)aligned_malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
    double *b_block = (double*)aligned_malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
    double *c_block = (double*)aligned_malloc(BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
    long block_num = m/BLOCK_SIZE;

    for(int j = 0; j < block_num; j++) {
        #pragma omp parallel for
        for(int i = 0; i < block_num; i++) {
            read_write(i, j, m, c_block, c, true);
            for(int p = 0; p < block_num; p++) {
                read_write(i, p, m, a_block, a, true);
                read_write(p, j, m, b_block, b, true);
                MMult1(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, a_block, b_block, c_block);
                read_write(i, j, m, c_block, c, false);
            }
        }
    }

    aligned_free(a_block);
    aligned_free(b_block);
    aligned_free(c_block);
}


int main(int argc, char** argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf(" Dimension       Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;

    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult3(m, n, k, a, b, c);
    }
    double time = t.toc();
    double flops = 2 * m * n * k * NREPEATS * sizeof(double) / 1e9 / time; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth = 4 * m * n * k * NREPEATS * sizeof(double) / 1e9 / time; // TODO: calculate from m, n, k, NREPEATS, time
    printf("%10ld %10f %10f %10f", p, time, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
