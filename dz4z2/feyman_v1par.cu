#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../runner/runner.h"

#define NUM_OF_GPU_THREADS 1024

int i4_ceiling(double x)
{
  int value = (int)x;
  if (value < x)
    value = value + 1;
  return value;
}

int i4_min(int i1, int i2)
{
  int value;
  if (i1 < i2)
    value = i1;
  else
    value = i2;
  return value;
}

__device__ float mpow(float a, int b)
{
  return a*a;
}

__device__ float potential(float a, float b, float c, float x, float y, float z)
{
  return 2.0 * (mpow(x / a / a, 2) + mpow(y / b / b, 2) + mpow(z / c / c, 2)) + 1.0 / a / a + 1.0 / b / b + 1.0 / c / c;
}

__device__ float r8_uniform_01(int *seed)
{
  int k;
  float r;

  k = *seed / 127773;

  *seed = 16807 * (*seed - k * 127773) - k * 2836;

  if (*seed < 0)
  {
    *seed = *seed + 2147483647;
  }
  r = (float)(*seed) * 4.656612875E-10;

  return r;
}

void timestamp(void)
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}

__global__ void kernel(float* err,int* n_inside, int ni, int nj, int nk, int a,int b, int c,int N, float stepsz, float h)
{
  //int numblocks = (N+NUM_OF_GPU_THREADS-1)/NUM_OF_GPU_THREADS;
  int seed = blockIdx.x + blockIdx.y * 2 + blockIdx.z * 4 + threadIdx.x * 8 + 4853694;
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  float x = ((float)(ni - i) * (-a) + (float)(i - 1) * a) / (float)(ni - 1);
  float y = ((float)(nj - j) * (-b) + (float)(j - 1) * b) / (float)(nj - 1);
  float z = ((float)(nk - k) * (-c) + (float)(k - 1) * c) / (float)(nk - 1);
  float w_exact;
  float steps_ave;
  float w;
  float us;
  float ut;
  float dx;
  float dy;
  float dz;
  float vh;
  float vs;
  float we;
  __shared__ float wt[NUM_OF_GPU_THREADS];
  //int steps;
  int trial;

  float chk = mpow(x / a, 2) + mpow(y / b, 2) + mpow(z / c, 2);

  if (1.0 >= chk)
  {
    w_exact = exp(mpow(x / a, 2) + mpow(y / b, 2) + mpow(z / c, 2) - 1.0);

    wt[threadIdx.x] = 0.0;
    //steps = 0;
    for (trial = threadIdx.x; trial < N; trial+=NUM_OF_GPU_THREADS)
    {
      float x1 = x;
      float x2 = y;
      float x3 = z;
      w = 1.0;
      chk = 0.0;
      while (chk < 1.0)
      {
        ut = r8_uniform_01(&seed);
        if (ut < 1.0 / 3.0)
        {
          us = r8_uniform_01(&seed) - 0.5;
          if (us < 0.0)
            dx = -stepsz;
          else
            dx = stepsz;
        }
        else
          dx = 0.0;

        ut = r8_uniform_01(&seed);
        if (ut < 1.0 / 3.0)
        {
          us = r8_uniform_01(&seed) - 0.5;
          if (us < 0.0)
            dy = -stepsz;
          else
            dy = stepsz;
        }
        else
          dy = 0.0;

        ut = r8_uniform_01(&seed);
        if (ut < 1.0 / 3.0)
        {
          us = r8_uniform_01(&seed) - 0.5;
          if (us < 0.0)
            dz = -stepsz;
          else
            dz = stepsz;
        }
        else
          dz = 0.0;

        vs = potential(a, b, c, x1, x2, x3);
        x1 = x1 + dx;
        x2 = x2 + dy;
        x3 = x3 + dz;

        //steps++;

        vh = potential(a, b, c, x1, x2, x3);

        we = (1.0 - h * vs) * w;
        w = w - 0.5 * h * (vh * we + vs * w);

        chk = mpow(x1 / a, 2) + mpow(x2 / b, 2) + mpow(x3 / c, 2);
      }
      wt[threadIdx.x] += w;
    }

    __syncthreads();
    for (int iter = blockDim.x >> 1 ; iter > 0; iter >>= 1) 
    {
      if ( threadIdx.x < iter) 
      {
        wt[threadIdx.x] += wt[threadIdx.x + iter];
      } 
      __syncthreads();
    }

    if (threadIdx.x == 0) 
    {
        wt[0] /= (float)N;
        atomicAdd(err,mpow(w_exact - wt[0], 2));
        atomicAdd(n_inside,1);
    }
  }
}

// print na stdout upotrebiti u validaciji paralelnog resenja
int main(int arc, char **argv)
{
  double a = 3.0;
  double b = 2.0;
  double c = 1.0;
  int dim = 3;
  double h = 0.001;
  int ni;
  int nj;
  int nk;
  double stepsz;
  int seed = 123456789;

  int N = atoi(argv[1]);
  timestamp();

  printf("A = %f\n", a);
  printf("B = %f\n", b);
  printf("C = %f\n", c);
  printf("N = %d\n", N);
  printf("H = %6.4f\n", h);

  stepsz = sqrt((double)dim * h);

  if (a == i4_min(i4_min(a, b), c))
  {
    ni = 6;
    nj = 1 + i4_ceiling(b / a) * (ni - 1);
    nk = 1 + i4_ceiling(c / a) * (ni - 1);
  }
  else if (b == i4_min(i4_min(a, b), c))
  {
    nj = 6;
    ni = 1 + i4_ceiling(a / b) * (nj - 1);
    nk = 1 + i4_ceiling(c / b) * (nj - 1);
  }
  else
  {
    nk = 6;
    ni = 1 + i4_ceiling(a / c) * (nk - 1);
    nj = 1 + i4_ceiling(b / c) * (nk - 1);
  }

  int numblocks = (N+NUM_OF_GPU_THREADS-1)/NUM_OF_GPU_THREADS;
  dim3 gridDim(ni*numblocks,nj,nk);
  dim3 blockDim(NUM_OF_GPU_THREADS);

  float err = 0;
  int n_inside = 0;

  float* globalErr;
  int* globalN_inside;
  cudaMalloc(&globalErr,sizeof(float));
  cudaMalloc(&globalN_inside,sizeof(int));
  cudaMemcpy(globalErr,&err,sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(globalN_inside,&n_inside,sizeof(int),cudaMemcpyHostToDevice);

  __runner__start();

  kernel<<<gridDim,blockDim>>>(globalErr,globalN_inside,ni,nj,nk,a,b,c,N,stepsz,h);

  __runner__stop();

  cudaMemcpy(&err,globalErr,sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(&n_inside,globalN_inside,sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(globalErr);
  cudaFree(globalN_inside);
  err = sqrt(err / (float)(n_inside));

  printf("\n\nRMS absolute error in solution = %e\n", err);
  timestamp();

  __runner__print();

  return 0;
}
