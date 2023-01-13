#include <stdio.h>
#include <cuda_runtime.h>

cudaEvent_t __runner__start__event;
cudaEvent_t __runner__end__event;
float __runner__time=0;

void __runner__start()
{
    cudaEventCreate(&__runner__start__event);
    cudaEventCreate(&__runner__end__event);
    cudaEventRecord(__runner__start__event);
}

void __runner__stop()
{
    cudaEventRecord(__runner__end__event);
    cudaEventSynchronize(__runner__end__event);
    cudaEventElapsedTime(&__runner__time, __runner__start__event, __runner__end__event);
}

void __runner__print()
{
    printf("Time: %f\n",__runner__time);
}