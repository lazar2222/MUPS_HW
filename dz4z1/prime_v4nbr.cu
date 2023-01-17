#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "../runner/runner.h"

#define NUM_OF_GPU_THREADS 1024

int nearestPowerOf2(int N)
{
    int a = log2(N);
 
    if (pow(2, a) == N)
        return N;
 
    return pow(2, a + 1);
}

double cpu_time(void)
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;

  return value;
}

__global__ void prime_number(int* globalTotal, int n)
{
  int i;
  int j;
  __shared__ volatile int total[NUM_OF_GPU_THREADS];

  total[threadIdx.x] = 1;

  i = 2 + (threadIdx.x * gridDim.x + blockIdx.x);
  if(i <= n)
  {
    for (j = 2; j < i; j++)
    {
      if ((i % j) == 0)
      {
        total[threadIdx.x] = 0;
        break;
      }
    }
  }
  else
  {
    total[threadIdx.x] = 0;
  }

  __syncthreads();
  
  for (int iter = blockDim.x >> 1 ; iter > 32; iter >>= 1) 
  {
    if ( threadIdx.x < iter) 
    {
      total[threadIdx.x] += total[threadIdx.x + iter];
    } 
    __syncthreads();
  } 
  if (threadIdx.x < 32) total[threadIdx.x] += total[threadIdx.x + 32];
  if (threadIdx.x < 16) total[threadIdx.x] += total[threadIdx.x + 16];
  if (threadIdx.x < 8)  total[threadIdx.x] += total[threadIdx.x + 8];
  if (threadIdx.x < 4)  total[threadIdx.x] += total[threadIdx.x + 4];
  if (threadIdx.x < 2)  total[threadIdx.x] += total[threadIdx.x + 2];
  if (threadIdx.x < 1)  atomicAdd(globalTotal,total[threadIdx.x] + total[threadIdx.x + 1]);
}

void timestamp(void)
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  len = strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}

void test(int n_lo, int n_hi, int n_factor);

int main(int argc, char *argv[])
{
  int n_factor;
  int n_hi;
  int n_lo;

  timestamp();
  printf("\n");
  printf("PRIME TEST\n");

  if (argc != 4)
  {
    n_lo = 1;
    n_hi = 131072;
    n_factor = 2;
  }
  else
  {
    n_lo = atoi(argv[1]);
    n_hi = atoi(argv[2]);
    n_factor = atoi(argv[3]);
  }

  test(n_lo, n_hi, n_factor);

  printf("\n");
  printf("PRIME_TEST\n");
  printf("  Normal end of execution.\n");
  printf("\n");
  timestamp();

  __runner__print();

  return 0;
}

void test(int n_lo, int n_hi, int n_factor)
{
  int i;
  int n;
  int primes;
  double ctime;

  printf("\n");
  printf("  Call PRIME_NUMBER to count the primes from 1 to N.\n");
  printf("\n");
  printf("         N        Pi          Time\n");
  printf("\n");

  n = n_lo;

  while (n <= n_hi)
  {
    int* globalTotal;
    primes = 0;

    int idealGrid = nearestPowerOf2(ceil(n/(float)NUM_OF_GPU_THREADS));
    if(idealGrid==1) idealGrid = 2;
    int idealBlock = nearestPowerOf2(n/(float)idealGrid);
    dim3 gridDim(idealGrid);
    dim3 blockDim(idealBlock);

    cudaMalloc(&globalTotal, sizeof(int));
    cudaMemcpy(globalTotal,&primes,sizeof(int),cudaMemcpyHostToDevice);

    ctime = cpu_time();

    __runner__start();

    prime_number<<<gridDim,blockDim>>>(globalTotal,n);

    __runner__stop();

    ctime = cpu_time() - ctime;

    cudaMemcpy(&primes,globalTotal,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(globalTotal);

    printf("  %8d  %8d  %14f\n", n, primes, ctime);
    n = n * n_factor;
  }

  return;
}
