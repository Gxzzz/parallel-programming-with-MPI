#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define SIZE_NOT_MATCH_ERROR 1

void manager(int manager_id, int n_proc, char *mat_file, char *vec_file) {
  int mat_r, mat_c, vec_size;

  // read vector size from file and broadcasts to all other processes
  FILE *fp = fopen(vec_file, "r");
  fread(&vec_size, sizeof(int), 1, fp);
  MPI_Bcast(&vec_size, 1, MPI_INT, manager_id, MPI_COMM_WORLD);

  // read vector data from file and broadcasts to all other processes
  double *vec = (double *)malloc(vec_size * sizeof(double));
  fread(vec, sizeof(double), vec_size, fp);
  MPI_Bcast(vec, vec_size, MPI_DOUBLE, manager_id, MPI_COMM_WORLD);

  // read matrix size from file
  fp = fopen(mat_file, "r");
  fread(&mat_r, sizeof(int), 1, fp);
  fread(&mat_c, sizeof(int), 1, fp);
  if (mat_c != vec_size) {
    printf("Size does not match\n");
    MPI_Abort(MPI_COMM_WORLD, SIZE_NOT_MATCH_ERROR);
  }
  // roadcasts matrix row count to all other processes, this
  // value is used as a finish TAG
  MPI_Bcast(&mat_r, 1, MPI_INT, manager_id, MPI_COMM_WORLD);

  // allocate an array to store matrix row
  double *mat_row = (double *)malloc(vec_size * sizeof(double));

  // distribute rows to workers
  int num = 0;
  for (int i = 0; i < min(mat_r, n_proc - 1); ++i) {
    fread(mat_row, sizeof(double), mat_c, fp);
    MPI_Send(mat_row, mat_c, MPI_DOUBLE, i, num++, MPI_COMM_WORLD);
  }

  // allocate an array to store the answer
  double *answer = (double *)malloc(mat_r * sizeof(double));

  // receive dot product from workers
  // and distribute rows to workers if there are unprocessed ones
  double dot_product;
  MPI_Status status;
  for (int _ = 0; _ < mat_r; ++_) {
    MPI_Recv(&dot_product, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    int worker_id = status.MPI_SOURCE;
    int row_id = status.MPI_TAG;

    answer[row_id] = dot_product;
    if (num < mat_r) {
      fread(mat_row, sizeof(double), mat_c, fp);
      MPI_Send(mat_row, mat_c, MPI_DOUBLE, worker_id, num++, MPI_COMM_WORLD);
    } else {
      MPI_Send(mat_row, mat_c, MPI_DOUBLE, worker_id, mat_r, MPI_COMM_WORLD);
    }
  }
  for (int i = 0; i < mat_r; ++i) {
    printf("%.1f ", answer[i]);
  }
  puts("");
}

void worker(int manager_id, int rank) {
  // receive vector size from the manager
  int mat_r, vec_size;
  MPI_Bcast(&vec_size, 1, MPI_INT, manager_id, MPI_COMM_WORLD);

  // receive vector data from the manager
  double *vec = (double *)malloc(vec_size * sizeof(double));
  MPI_Bcast(vec, vec_size, MPI_DOUBLE, manager_id, MPI_COMM_WORLD);

  // receive matrix row count from the manager, which is a signal
  // for finishing job
  MPI_Bcast(&mat_r, 1, MPI_INT, manager_id, MPI_COMM_WORLD);

  // In case #processes > #rows, some workers do nothing
  if (rank >= mat_r)
    return;

  // allocate an array to store matrix row
  double *mat_row = (double *)malloc(vec_size * sizeof(double));
  MPI_Status status;

  while (1) {
    MPI_Recv(mat_row, vec_size, MPI_DOUBLE, manager_id, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    if (status.MPI_TAG == mat_r)  break;

    double dot_product = 0;
    for (int i = 0; i < vec_size; ++i) {
      dot_product += vec[i] * mat_row[i];
    }
    MPI_Send(&dot_product, 1, MPI_DOUBLE, manager_id, status.MPI_TAG, MPI_COMM_WORLD);
  }
}

int main(int argc, char *argv[]) {
  int rank, p;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == p - 1) {
    manager(p - 1, p, argv[1], argv[2]);
  } else {
    worker(p - 1, rank);
  }
  MPI_Finalize();
  return 0;
}
