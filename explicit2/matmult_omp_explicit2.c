/* 
   Matrix multiplication example

   OpenMP version, explicit domain decomposition with reduction clause

   Jim Teresco, CS 338, Williams College, CS 341, Mount Holyoke College
   Sun Feb 23 18:54:41 EST 2003

   Updated for CSIS-335, Siena College, Fall 2021
*/

/* header files needed for printf, gettimeofday, struct timeval */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

/* header file for our own timer.c function diffgettime */
#include "timer.h"

/* we will multiply square matrices, how big? */
#define SIZE 1500

/* our matrices */
double a[SIZE][SIZE], b[SIZE][SIZE], c[SIZE][SIZE];

/* function to compute the result of row row in c */
void do_row(int row) {
  int col, k;

  for (col=0; col<SIZE; col++) {
    
    /* initialize entry */
    c[row][col] = 0;
    
    /* perform dot product */
    for(k=0; k<SIZE; k++) {
      c[row][col] = c[row][col] + a[row][k]*b[k][col];
    }
  }
}

/* this function will be called by each thread created.  Note that we
   could instead have put all of this code in a block under the
   #pragma omp parallel, but all of the local variables would have 
   had to be included in a private clause.
*/
void worker() {
  int thread_num, num_threads, row, myrows, extrarows, startrow, finishrow;

  /* query OpenMP for number of threads and thread id */
  thread_num = omp_get_thread_num();
  num_threads = omp_get_num_threads();

  /* some extra computation to deal with the fact that the number of
     threads might not be evenly divide the number of rows -- we will
     assign an extra row, as necessary, to higher numbered threads */
  myrows = SIZE/num_threads;
  extrarows = SIZE%num_threads;

  /* start out with the assumption that none of our predecessors have
     gotten an extra row */
  startrow = myrows*thread_num;

  /* see if this thread or any of its predecessors have an extra row */
  if (extrarows >= num_threads-thread_num) {
    /* compensate for extra rows in predecessors */
    startrow += extrarows - (num_threads-thread_num);
    /* this thread gets an extra row, too */
    myrows++;
  }
  
  /* now that we know our real starting row and number of rows,
     we can figure our last row */
  finishrow = startrow + myrows -1;
  
  printf("Worker %d will compute %d rows: %d-%d\n", thread_num, myrows,
	 startrow, finishrow);
  
  for (row=startrow; row<=finishrow; row++) 
    do_row(row);
}

/* it's a simple program for now, we'll just put everything in main */
int main(int argc, char *argv[]) {

  /* counters */
  int i, j, k;
  double sum;

  /* to pass to gettimeofday to get wall clock times */
  struct timeval start, stop;

  /* initialize and allocate matrices, just fill with junk */
  gettimeofday(&start, NULL);
  for (i=0; i<SIZE; i++) {
    for (j=0; j<SIZE; j++) {
      a[i][j] = i+j;
      b[i][j] = i-j;
    }
  }
  gettimeofday(&stop, NULL);
  printf("Initialization took: %f seconds\n", diffgettime(start,stop));
  
  gettimeofday(&start, NULL);
  /* matrix-matrix multiply */
#pragma omp parallel
  worker();  /* this is called by each thread */

  /* there is an implied barrier here -- the master thread cannot continue
     until it and all other threads have completed the worker() call. */

  gettimeofday(&stop, NULL);
  printf("Multiplication took: %f seconds\n", diffgettime(start,stop));
  
  /* Usually this is here to make sure the optimizing compiler doesn't
     get any big ideas about "optimizing" code away completely, but in
     this example we actually parallelize it with a reduction clause */
  sum=0;
#pragma omp parallel for private(i,j) reduction(+:sum)
  for (i=0; i<SIZE; i++) {
    for (j=0; j<SIZE; j++) {
      sum += c[i][j];
    }
  }

  printf("Sum of elements of c=%f\n", sum);

  return 0;
}
