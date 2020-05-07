#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define block_low(i, n, p) ((i) * (n) / (p))
#define block_high(i, n, p) (block_low(i + 1, n, p) - 1)

void print_root(int l, int r, int **root, int n) {
  int new_l = l + n - 1 - r;
  printf("Root of tree spanning %d-%d is %d\n", l, r, root[new_l][r]);
  if (l <= root[new_l][r] - 1)
    print_root(l, root[new_l][r] - 1, root, n);
  if (r >= root[new_l][r] + 1)
    print_root(root[new_l][r] + 1, r, root, n);
}

int main(int argc, char *argv[]) {
  int rank, n_proc;
  MPI_Status status;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  FILE *fp = fopen(argv[1], "r");
  int n;
  fscanf(fp, "%d", &n);
  float *p = (float *)malloc(n * sizeof(float));
  for (int i = 0; i < n; ++i) {
    fscanf(fp, "%f", &p[i]);
  }

  float *val_storage = (float *)malloc(n * n * sizeof(float));
  float **val = (float **)malloc(n * sizeof(float *));
  for (int i = 0; i < n; ++i) val[i] = val_storage + i * n;
  int *rt_storage = (int *)malloc(n * n * sizeof(int));
  int **rt = (int **)malloc(n * sizeof(int *));
  for (int i = 0; i < n; ++i) rt[i] = rt_storage + i * n;

  float *presum = (float *)malloc((n + 1) * sizeof(float));
  presum[0] = 0;
  for (int i = 1; i <= n; ++i) {
    presum[i] = presum[i - 1] + p[i - 1];
  }

  // auxiliary arrays for MPI_Allgatherv
  int *recvcounts = (int *)malloc(n_proc * sizeof(int));
  int *displs = (int *)malloc(n_proc * sizeof(int));

  /*
   * The original table is of this form:
   *  x x x x x
   *  0 x x x x
   *  0 0 x x x
   *  0 0 0 x x
   *  0 0 0 0 x
   * Put all the elements down so that the original diagonal elements are on the same row,
   * then the table should be of this form:
   *  0 0 0 0 x
   *  0 0 0 x x
   *  0 0 x x x
   *  0 x x x x
   *  x x x x x
   * In this way, the table is filled in a bottom-up way, in each outer iteration, one
   * row is filled and the task can be distributed to different processes.
   * */
  for (int r = n - 1; r >= 0; --r) {
    int st = block_low(rank, r + 1, n_proc);
    int ed = block_high(rank, r + 1, n_proc);
    int *rt_partial = (int *)malloc((ed - st + 1) * sizeof(int));
    float *val_partial = (float *)malloc((ed - st + 1) * sizeof(float));
    for (int c = n - 1 - r + st; c <= n - 1 - r + ed; ++c) {
      int rr = c - n + r + 1;
      float best_val = 1e10;
      int best_rt = -1;
      for (int k = rr; k <= c; ++k) {
        float tmp = p[k];
        if (k - 1 >= rr)  tmp += val[rr + (n - 1 - (k - 1))][k - 1] + presum[k] - presum[rr];
        if (k + 1 <= c) tmp += val[k + 1 + (n - 1 - c)][c] + presum[c + 1] - presum[k + 1];
        if (tmp < best_val) best_val = tmp, best_rt = k;
      }
      rt_partial[c - (n - 1 - r + st)] = best_rt;
      val_partial[c - (n - 1 - r + st)] = best_val;
    }
    for (int i = 0; i < n_proc; ++i) {
      recvcounts[i] = block_high(i, r + 1, n_proc) - block_low(i, r + 1, n_proc) + 1;
      displs[i] = block_low(i, r + 1, n_proc);
    }
    MPI_Allgatherv(rt_partial, ed - st + 1, MPI_INT, rt[r] + n - 1 - r, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgatherv(val_partial, ed - st + 1, MPI_FLOAT, val[r] + n - 1 - r, recvcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);
    free(rt_partial);
    free(val_partial);
  }
  
  if (rank == n_proc - 1)
    print_root(0, n - 1, rt, n);
  return 0;
}
