#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#define block_low(i, n, p) ((i) * (n) / (p))
#define block_high(i, n, p) (block_low(i + 1, n, p) - 1)
#define block_owner(idx, n, p) (((p) * ((idx) + 1) - 1) / (n))
#define min(a, b) (((a) < (b)) ? (a) : (b))

int main(int argc, char *argv[]) {
  int rank, p;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n;
  FILE *fp;
  // process p-1 reads matrix size from file and
  // broadcasts to all other processes
  if (rank == p - 1) {
    fp = fopen(argv[1], "r");
    fread(&n, sizeof(int), 1, fp);
  }
  MPI_Bcast(&n, 1, MPI_INT, p - 1, MPI_COMM_WORLD);

  // each process allocates memory for its rows
  int st = block_low(rank, n, p);
  int ed = block_high(rank, n, p);
  int m = ed - st + 1;
  int *storage = (int *)malloc(m * n * sizeof(int));
  int **mt = (int **)malloc(m * sizeof(int *));
  for (int i = 0; i < m; ++i)
    mt[i] = &storage[i * n];

  // process p-1 reads data from file and
  // distributes it to all other processes
  if (rank == p - 1) {
    for (int rk = 0; rk < p - 1; ++rk) {
      int n_rows = block_high(rk, n, p) - block_low(rk, n, p) + 1;
      fread(storage, sizeof(int), n_rows * n, fp);
      MPI_Send(storage, n_rows * n, MPI_INT, rk, 0, MPI_COMM_WORLD);
    }
    fread(storage, sizeof(int), m * n, fp);
  } else {
    MPI_Recv(storage, m * n, MPI_INT, p - 1, 0, MPI_COMM_WORLD, &status);
  }

  // Floyd
  int *tmp = (int *)malloc(n * sizeof(int));
  for (int k = 0; k < n; ++k) {
    int k_owner = block_owner(k, n, p);
    if (k_owner == rank) {
      memcpy(tmp, mt[k - st], n * sizeof(n));
    }
    MPI_Bcast(tmp, n, MPI_INT, k_owner, MPI_COMM_WORLD);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        mt[i][j] = min(mt[i][k] + tmp[j], mt[i][j]);
      }
    }
  }
  int flag;
  if (rank)
    MPI_Recv(&flag, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%d%c", mt[i][j], " \n"[j == n - 1]);
    }
  }
  if (rank < p - 1)
    MPI_Send(&flag, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
  free(storage);
  free(mt);
  free(tmp);
  MPI_Finalize();
  return 0;
}
