#include <stdio.h>
#include <mpi.h>

double __runner__start__time;
double __runner__end__time;
double __runner__time=0;

void __runner__start()
{
    __runner__start__time = MPI_Wtime();
}

void __runner__stop()
{
    __runner__end__time = MPI_Wtime();
    __runner__time += __runner__end__time - __runner__start__time;
}

void __runner__print()
{
    printf("Time: %lf\n",__runner__time);
}