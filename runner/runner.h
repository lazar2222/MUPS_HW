#include <stdio.h>
#include <omp.h>

double __runner__start__time;
double __runner__end__time;
double __runner__time=0;

void __runner__start()
{
    __runner__start__time = omp_get_wtime();
}

void __runner__stop()
{
    __runner__end__time = omp_get_wtime();
    __runner__time += __runner__end__time - __runner__start__time;
}

void __runner__print()
{
    printf("Time: %lf\n",__runner__time);
}