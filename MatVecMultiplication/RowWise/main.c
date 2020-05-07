#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define block_low(i, mat_c, p) ((i) * (mat_c) / (p))
#define block_high(i, mat_c, p) (block_low(i + 1, mat_c, p) - 1)
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define SIZE_NOT_MATCH_ERROR 1

int main(int argc, char *argv[]) {
  int rank, p;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int mat_r, mat_c, vec_size;
  FILE *fp;
  // process p-1 reads matrix size from file and
  // broadcasts to all other processes
  if (rank == p - 1) {
    fp = fopen(argv[1], "r");
    fread(&mat_r, sizeof(int), 1, fp);
    fread(&mat_c, sizeof(int), 1, fp);
  }
  MPI_Bcast(&mat_r, 1, MPI_INT, p - 1, MPI_COMM_WORLD);
  MPI_Bcast(&mat_c, 1, MPI_INT, p - 1, MPI_COMM_WORLD);

  // each process allocates memory for its rows
  int st = block_low(rank, mat_r, p);
  int ed = block_high(rank, mat_r, p);
  int local_r = ed - st + 1;
  double *mat_storage = (double *)malloc(local_r * mat_c * sizeof(double));
  double **mat = (double **)malloc(local_r * sizeof(double *));
  for (int i = 0; i < local_r; ++i)
    mat[i] = &mat_storage[i * mat_c];

  // process p-1 reads matrix data from file and
  // distributes it to all other processes
  if (rank == p - 1) {
    for (int rk = 0; rk < p - 1; ++rk) {
      int n_rows = block_high(rk, mat_r, p) - block_low(rk, mat_r, p) + 1;
      fread(mat_storage, sizeof(double), n_rows * mat_c, fp);
      MPI_Send(mat_storage, n_rows * mat_c, MPI_DOUBLE, rk, 0, MPI_COMM_WORLD);
    }
    fread(mat_storage, sizeof(double), local_r * mat_c, fp);
    fclose(fp);
  } else {
    MPI_Recv(mat_storage, local_r * mat_c, MPI_DOUBLE, p - 1, 0, MPI_COMM_WORLD, &status);
  }
 
  // process p-1 reads vector size from file and
  // broadcasts to all other processes 
  if (rank == p - 1) {
    fp = fopen(argv[2], "r");
    fread(&vec_size, sizeof(int), 1, fp);
    if (vec_size != mat_c) {
      printf("Size does not match\n");
      MPI_Abort(MPI_COMM_WORLD, SIZE_NOT_MATCH_ERROR);
    }
  }
  MPI_Bcast(&vec_size, 1, MPI_INT, p - 1, MPI_COMM_WORLD);
  double *vec = (double *)malloc(vec_size * sizeof(double));

  // process p-1 reads vec data from file and
  // distributes it to all other processes
  if (rank == p - 1) {
    fread(vec, sizeof(double), vec_size, fp);
    fclose(fp);
  }
  MPI_Bcast(vec, vec_size, MPI_DOUBLE, p - 1, MPI_COMM_WORLD);
  
  double *res_partial = (double *)malloc(local_r * sizeof(double));
  double *res = (double *)malloc(mat_r * sizeof(double));
  for (int i = 0; i < local_r; ++i) {
    res_partial[i] = 0;
    for (int c = 0; c < mat_c; ++c) {
      res_partial[i] += mat[i][c] * vec[c];
    }
  }
  
  int *recvcounts = (int *)malloc(p * sizeof(int));
  int *displs = (int *)malloc(p * sizeof(int));
  for (int i = 0; i < p; ++i) {
    recvcounts[i] = block_high(i, mat_r, p) - block_low(i, mat_r, p) + 1;
    displs[i] = block_low(i, mat_r, p);
  }
  MPI_Allgatherv(res_partial, local_r, MPI_DOUBLE, res, recvcounts,
      displs, MPI_DOUBLE, MPI_COMM_WORLD);
  int flag;
  if (rank)
    MPI_Recv(&flag, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
  printf("rank=%d: ", rank);
  for (int i = 0; i < mat_r; ++i) {
    printf("%.1f%c", res[i], " \n"[i == mat_r - 1]);
  }
  if (rank < p - 1)
    MPI_Send(&flag, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
  free(mat_storage);
  free(mat);
  MPI_Finalize();
  return 0;
}
