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
  int N = 100;

  int* vec = (int*)malloc(N*sizeof(int));
  int *splitters = (int*)malloc(p*(p-1)*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int dist = (int)round(((double)N)/((double)(p-1)));
  int st = (int)ceil(((double)dist)/2.0);
  int *sample = (int*)malloc((p-1)*sizeof(int));

  int j = 0;
  for(int i = st; i < N; i += dist) {
      sample[j++] = vec[i];
  }

  printf("%d: st: %d  dist: %d\n", rank, st, dist);

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  MPI_Gather(sample, p-1, MPI_INT, splitters, p-1, MPI_INT, root, MPI_COMM_WORLD);


  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  if(rank == root) {
      for(int i = 0; i < p*(p-1); i++) {
          printf("%d:: %d\n", i, splitters[i]);
      }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // root process broadcasts splitters to all other processes

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

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data

  // do a local sort of the received data

  // every process writes its result to a file

  free(vec);
  free(sample);
  MPI_Finalize();
  return 0;
}
