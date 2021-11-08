/* 
   Matrix multiplication example

   OpenMP version, bag of tasks

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

/* it's a simple program for now, we'll just put everything in main */
int main(int argc, char *argv[]) {

  /* counters */
  int i, j, k;
  double sum;

  /* to pass to gettimeofday to get wall clock times */
  struct timeval start, stop;

  /* our bag of tasks - each row is a task */
  int next_avail_task = 0;
  int current_task;

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
#pragma omp parallel private(current_task) shared(next_avail_task)
  {
    /* mutual exclution on next_avail_task */
#pragma omp critical(mutex)
    current_task = next_avail_task++;

    /* process rows from the bag of tasks */
    while (current_task < SIZE) {
      do_row(current_task);

      /* mutual exclusion on next_avail_task */
#pragma omp critical(mutex)
    current_task = next_avail_task++;
    }
  } /* end of parallel block */

  /* there is an implied barrier here -- the master thread cannot continue
     until it and all other threads have completed the parallel block. */

  gettimeofday(&stop, NULL);
  printf("Multiplication took: %f seconds\n", diffgettime(start,stop));
  
  /* This is here to make sure the optimizing compiler doesn't
     get any big ideas about "optimizing" code away completely */
  sum=0;
  for (i=0; i<SIZE; i++) {
    for (j=0; j<SIZE; j++) {
      sum += c[i][j];
    }
  }
  printf("Sum of elements of c=%f\n", sum);

  return 0;
}
