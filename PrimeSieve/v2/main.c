#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#define max(a, b) ((a) > (b) ? (a) : (b))

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
  char *marked = (char *)calloc(n + 1, 1);

  int n_root = sqrt(n);
  // sieve prime numbers in [2, n_root]
  for (int p = 2; p <= n_root / p; ++p) {
    if (marked[p])  continue;
    for (int i = p * p; i <= n_root; i += p) {
      marked[i] = 1;
    }
  }
  
  for (int p = 2, cnt = 0; p <= n_root; ++p) {
    if (marked[p])  continue;
    if (cnt % n_proc == rank) {
      int start = max(p * p, (n_root + 1 + p - 1) / p * p);
      for (int i = start; i <= n; i += p) {
        marked[i] = 1;
      }
    }
  }
  
  if (rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, marked, n + 1, MPI_CHAR, MPI_LOR, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(marked, NULL, n + 1, MPI_CHAR, MPI_LOR, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    int cnt = 0;
    for (int i = 2; i <= n; ++i) {
      cnt += !marked[i];
    }
    printf("%d\n", cnt);
    printf("time elpased: %.1f sec\n", MPI_Wtime() - start_time);
  }
  MPI_Finalize();
  return 0;
}
