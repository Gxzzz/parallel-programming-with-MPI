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

  // each process allocates memory for its matrix cols
  // and vector part
  int st = block_low(rank, mat_c, p);
  int ed = block_high(rank, mat_c, p);
  int local_c = ed - st + 1;
  double *mat_storage = (double *)malloc(mat_r * local_c * sizeof(double));
  double **mat = (double **)malloc(mat_r * sizeof(double *));
  for (int i = 0; i < mat_r; ++i)
    mat[i] = &mat_storage[i * local_c];
  double *vec = (double *)malloc(local_c * sizeof(double));

  // process p-1 reads matrix data from file and
  // distributes it to all other processes
  if (rank == p - 1) {
    for (int i = 0; i < mat_r; ++i) {
      for (int rk = 0; rk < p - 1; ++rk) {
        int n_cols = block_high(rk, mat_c, p) - block_low(rk, mat_c, p) + 1;
        fread(mat[i], sizeof(double), n_cols, fp);
        MPI_Send(mat[i], n_cols, MPI_DOUBLE, rk, i, MPI_COMM_WORLD);
      }
      fread(mat[i], sizeof(double), local_c, fp);
    }
    fclose(fp);
  } else {
    for (int i = 0; i < mat_r; ++i) {
      MPI_Recv(mat[i], local_c, MPI_DOUBLE, p - 1, i, MPI_COMM_WORLD, &status);
    }
  }
 
  // process p-1 reads vector data from file and
  // broadcasts to all other processes 
  if (rank == p - 1) {
    fp = fopen(argv[2], "r");
    fread(&vec_size, sizeof(int), 1, fp);
    if (vec_size != mat_c) {
      printf("Size does not match\n");
      MPI_Abort(MPI_COMM_WORLD, SIZE_NOT_MATCH_ERROR);
    }
    for (int rk = 0; rk < p - 1; ++rk) {
      int n_cols = block_high(rk, mat_c, p) - block_low(rk, mat_c, p) + 1;
      fread(vec, sizeof(double), n_cols, fp);
      MPI_Send(vec, n_cols, MPI_DOUBLE, rk, 0, MPI_COMM_WORLD);
    }
    fread(vec, sizeof(double), local_c, fp);
    fclose(fp);
  } else {
    MPI_Recv(vec, local_c, MPI_DOUBLE, p - 1, 0, MPI_COMM_WORLD, &status);
  }
  
  double *res = (double *)malloc(mat_r * sizeof(double));
  for (int r = 0; r < mat_r; ++r) {
    res[r] = 0;
    for (int c = 0; c < local_c; ++c) {
      res[r] += mat[r][c] * vec[c];
    }
  }
  
  MPI_Allreduce(MPI_IN_PLACE, res, mat_r, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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
