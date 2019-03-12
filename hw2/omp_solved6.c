/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

float dotprod ()
{
int i,tid;
float sum = 0.0;

tid = omp_get_thread_num();
// no need for reduction as our shared sum is in main function, not visible in this scope
// here sum is private variable and hence needs to be done without reduction
#pragma omp for
  for (i=0; i < VECLEN; i++)
    {
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }

// also need to return sum computed by this thread
return sum;
}


int main (int argc, char *argv[]) {
int i;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

#pragma omp parallel shared(sum)
{
  // this temporary variable is private holding temporary value
  // so that critical write can be done later on and dotprod can execute in parallel
  float temp = dotprod();

  // since sum is shared variable, multiple threads may try to write to it at the same time
  // hence making it critical so that its updated only by one thread at a time
  #pragma omp critical
    sum += temp;
}

printf("Sum = %f\n",sum);

}
