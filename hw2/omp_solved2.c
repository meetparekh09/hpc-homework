/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug.
* AUTHOR: Blaise Barney
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
    int nthreads;

    // No need of initialization outside parallel region as these variables are not shared by parallel region
    // int i, tid;
    // float total;

/*** Spawn parallel region ***/
#pragma omp parallel
  {
  /* Obtain thread number */
  int tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
  float total = 0.0;

  // This parallel for is misguiding as it does not initialize new parallel implementation of for
  // but redistributes the execution of this for loop amongst already spawned threads
  // this means that every thread having its own version of total would be modified which gives inconsistent result
  // However if only one total needs to be computed then, an alternative solution is given below.

  // #pragma omp for schedule(dynamic,10)
  for (int i=0; i<1000000; i++)
     total = total + i*1.0;

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/


  // Solution 2
  float total = 0.0f;

/*** Spawn parallel region ***/
#pragma omp parallel
{
/* Obtain thread number */
int tid = omp_get_thread_num();
/* Only master thread does this */
if (tid == 0) {
  nthreads = omp_get_num_threads();
  printf("Number of threads = %d\n", nthreads);
  }
printf("Thread %d is starting...\n",tid);

#pragma omp barrier

#pragma omp for schedule(dynamic,10) reduction (+:total)
for (int i=0; i<1000000; i++)
   total = total + i*1.0;

printf ("Thread %d is done! Total= %e\n",tid,total);

} /*** End of parallel region ***/

}
