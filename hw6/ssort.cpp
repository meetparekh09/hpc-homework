// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  int root = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 1000000;

  int* vec = (int*)malloc(N*sizeof(int));
  int *samples;
  int *splitters = (int*)malloc((p-1)*sizeof(int));
  int *sdispls = (int*)malloc(p*sizeof(int));
  int *scounts = (int*)malloc(p*sizeof(int));
  int *rcounts = (int*)malloc(p*sizeof(int));
  int *rdispls = (int*)malloc(p*sizeof(int));
  int *bucket;

  if(rank == root)
    samples = (int*)malloc(p*(p-1)*sizeof(int));
  else
    samples = NULL;
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }

  double t1 = MPI_Wtime();
  // sort locally
  std::sort(vec, vec+N);

  // MPI_Barrier(MPI_COMM_WORLD);
  //
  // for(int i = 0; i < N; i++) {
  //     printf("%d:: %d => %d\n", rank, i, vec[i]);
  // }
  //
  // MPI_Barrier(MPI_COMM_WORLD);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int dist = (int)round(((double)N)/((double)(p)));
  int st = dist-1;
  int *sample = (int*)malloc((p-1)*sizeof(int));

  int j = 0;
  for(int i = st; i < N; i += dist) {
      sample[j++] = vec[i];
      if(j == p-1) break;
  }
  // printf("%d %d\n", rank, j);


  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  MPI_Gather(sample, p-1, MPI_INT, samples, p-1, MPI_INT, root, MPI_COMM_WORLD);


  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  if(rank == root) {
      std::sort(samples, samples+(p*(p-1)));
      dist = p-1;
      st = dist-1;

      int j = 0;
      for(int i = st; i < p*p-1; i += dist) {
          splitters[j++] = samples[i];
          if(j == p-1) break;
      }
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast(splitters, p-1, MPI_INT, root, MPI_COMM_WORLD);

  // if(rank == root) {
  // printf("\n\nSplitters\n");
  // for(int i = 0; i < p-1; i++) {
  //     printf("%d\n", splitters[i]);
  // }
  // printf("\n\n");
  // }
  // MPI_Barrier(MPI_COMM_WORLD);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  sdispls[0] = 0;
  for(int i = 1; i < p; i++) {
      sdispls[i] = std::lower_bound(vec, vec+N, splitters[i-1]) - vec;
      if(i == 1) scounts[i-1] = sdispls[i] - sdispls[i-1];
      else scounts[i-1] = sdispls[i] - sdispls[i-1];
  }
  scounts[p-1] = N - sdispls[p-1];

  int total = 0;
  for(int i = 0; i < p; i++) {
      total += scounts[i];
  }

  // MPI_Barrier(MPI_COMM_WORLD);
  //
  // printf("rank to_rank sdispls scounts\n");
  // for(int i = 0; i < p; i++) {
  //     printf("%d %d %d %d\n", rank, i, sdispls[i], scounts[i]);
  // }
  //
  // MPI_Barrier(MPI_COMM_WORLD);

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, MPI_COMM_WORLD);
  total = 0;
  for(int i = 0; i < p; i++) {
      if(i == 0) rdispls[i] = 0;
      else rdispls[i] = rdispls[i-1] + rcounts[i-1];
      total += rcounts[i];
  }

  // MPI_Barrier(MPI_COMM_WORLD);
  //
  // printf("\n\n");
  // printf("rank from_rank rdispls rcounts\n");
  // for(int i = 0; i < p; i++) {
  //     printf("%d %d %d %d\n", rank, i, rdispls[i], rcounts[i]);
  // }
  // printf("\n\n");
  //
  // MPI_Barrier(MPI_COMM_WORLD);

  // printf("Rank %d :: Count %d\n", rank, total);

  bucket = (int*)malloc(total*sizeof(int));

  MPI_Alltoallv(vec, scounts, sdispls, MPI_INT, bucket, rcounts, rdispls, MPI_INT, MPI_COMM_WORLD);


  // do a local sort of the received data
  std::sort(bucket, bucket+total);

  double t2 = MPI_Wtime();

  if(rank == root) {
      printf("Time taken for N = %d and %d Cores :: %f\n", N, p, t2-t1);
  }


  // every process writes its result to a file
  FILE* fd = NULL;
  char filename[256];
  snprintf(filename, 256, "output%02d.txt", rank);
  fd = fopen(filename,"w+");

  if(NULL == fd) {
    printf("Error opening file \n");
    return 1;
  }

  for(int i = 0; i < total; i++)
    fprintf(fd, "  %d\n", bucket[i]);

  fclose(fd);

  free(vec);
  free(sample);
  free(samples);
  free(splitters);
  free(sdispls);
  free(scounts);
  free(rcounts);
  free(rdispls);
  free(bucket);
  MPI_Finalize();
  return 0;
}
