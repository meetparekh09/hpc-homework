#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n, int threads) {
    int partition;
    int nthreads;

    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        if(tid == 0) {
            nthreads = omp_get_num_threads();
            // printf("Number of threads :: %d\n", nthreads);
            partition = n/nthreads;
        }

        #pragma omp barrier

        long lower = tid*partition;
        long upper;
        if(tid == nthreads - 1)
            upper = n;
        else
            upper = (tid+1)*partition;


        prefix_sum[lower] = 0;
        for(long i = lower+1; i < upper; i++) {
            prefix_sum[i] = prefix_sum[i-1] + A[i-1];
        }

        #pragma omp barrier
    }

    for(long i = partition; i < n; i+=partition) {
        if(i + partition > n)
            continue;
        long j = i;
        long upper;
        if(i + 2*partition > n)
            upper = n;
        else
            upper = i+partition;
        while(j < upper) {
            prefix_sum[j++] += prefix_sum[i-1] + A[i-1];
        }
    }
}

int main() {
  long n[6] = {100000, 1000000, 10000000, 100000000, 1000000000, 10000000000};
  int threads[6] = {10, 15, 20, 25, 30, 40};
  for(int j = 0; j < 6; j++) {
      long N = n[j];
      long* A = (long*) malloc(N * sizeof(long));
      long* B0 = (long*) malloc(N * sizeof(long));
      long* B1 = (long*) malloc(N * sizeof(long));
      for (long i = 0; i < N; i++) A[i] = rand();


      double tt = omp_get_wtime();
      scan_seq(B0, A, N);
      printf("N = %ld, sequential-scan = %fs\n", N, omp_get_wtime() - tt);

      for(int k = 0; k < 6; k++) {
          tt = omp_get_wtime();
          scan_omp(B1, A, N, threads[k]);
          printf("N = %ld, Num Threads = %d, parallel-scan   = %fs, ", N, threads[k], omp_get_wtime() - tt);
          long err = 0;
          for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
          printf("error = %ld\n", err);
      }

      // long err = 0;
      // for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
      // printf("error = %ld\n", err);

      free(A);
      free(B0);
      free(B1);
  }

  return 0;
}
