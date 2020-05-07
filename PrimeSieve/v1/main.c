#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#define max(a, b) ((a) > (b) ? (a) : (b))
#define block_low(i, n, n_proc) ((i) * (n) / (n_proc))
#define block_high(i, n, n_proc) block_low(i + 1, n, n_proc) - 1

int main(int argc, char *argv[]) {
  int rank, n_proc;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double start_time = MPI_Wtime();
  int n = 100000;
  if (argc > 1) {
    n = atoi(argv[1]);
  }
  // sieve prime numbers from 2 to n
  // current process should proceed interval [low, high]
  int low = 2 + block_low(rank, n - 1, n_proc);
  int high = 2 + block_high(rank, n - 1, n_proc);
  // process 0 must cover interval [2, sqrt(n)]
  if (rank == 0 && high < (int)sqrt(n)) {
    fprintf(stderr, "Too many processes.");
    exit(1);
  }
  
  char *marked = (char *)calloc(high - low + 1, 1);
  int prime = 2;
  while (prime <= n / prime) {
    // start is the least number that is greater or equal to
    // prime^2 and low, and divides prime.
    int start = max(prime * prime, (low + prime - 1) / prime * prime);
    for (int i = start; i <= high; i += prime) {
      marked[i - low] = 1;
    }
    if (rank == 0) {
      while (1) {
        ++prime;
        if (prime > n / prime || marked[prime - low] == 0)
          break;
      }
    }
    MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  int cnt = 0;
  for (int i = low; i <= high; ++i) {
    cnt += !marked[i - low];
  }
  if (rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &cnt, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&cnt, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  }
  if (rank == 0) {
    printf("%d\n", cnt);
    printf("time elapsed: %.1f sec\n", MPI_Wtime() - start_time);
  }
  free(marked);
  MPI_Finalize();
  return 0;
}
