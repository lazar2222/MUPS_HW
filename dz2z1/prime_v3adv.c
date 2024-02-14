#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "../runner/runner.h"

#define min(x,y) ((x)<(y)?(x):(y))
#define MASTER 0
#define MASTER_ONLY if(rank == MASTER)

double cpu_time(void)
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;

  return value;
}

int prime_number(int n, int rank, int size)
{
  int i;
  int j;
  int prime;
  int total;

  total = 0;

  for (int x = 2 + rank * 16; x <= n; x +=  16 * size)
  {
    for (i = x; i < x + 16 && i<=n; i++)
    {
      prime = 1;
      for (j = 2; j < i; j++)
      {
        if ((i % j) == 0)
        {
          prime = 0;
          break;
        }
      }
      total = total + prime;
    }
  }
  return total;
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

void test(int n_lo, int n_hi, int n_factor, int rank, int size);

int main(int argc, char *argv[])
{
  int n_factor;
  int n_hi;
  int n_lo;
  int rank;
  int size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(size > 4)
  {
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  
  MASTER_ONLY
  {
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
  }

  int arr[] = {n_lo,n_hi,n_factor};
  MPI_Bcast(arr, 3, MPI_INT, MASTER, MPI_COMM_WORLD);
  n_lo = arr[0];
  n_hi = arr[1];
  n_factor = arr[2];

  test(n_lo, n_hi, n_factor, rank, size);

  MASTER_ONLY
  {
    printf("\n");
    printf("PRIME_TEST\n");
    printf("  Normal end of execution.\n");
    printf("\n");
    timestamp();

    __runner__print();
  }

  MPI_Finalize();

  return 0;
}

void test(int n_lo, int n_hi, int n_factor, int rank, int size)
{
  int i;
  int n;
  int primes;
  int tmpPrimes;
  double ctime;

  MASTER_ONLY
  {
    printf("\n");
    printf("  Call PRIME_NUMBER to count the primes from 1 to N.\n");
    printf("\n");
    printf("         N        Pi          Time\n");
    printf("\n");
  }

  n = n_lo;

  while (n <= n_hi)
  {
    MASTER_ONLY
    {
      ctime = cpu_time();

      __runner__start();
    }

    tmpPrimes = prime_number(n, rank, size);
    MPI_Reduce(&tmpPrimes, &primes, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

    MASTER_ONLY
    {
      __runner__stop();

      ctime = cpu_time() - ctime;

      printf("  %8d  %8d  %14f\n", n, primes, ctime);
    }
    n = n * n_factor;
  }

  return;
}
