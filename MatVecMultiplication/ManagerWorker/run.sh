mpicc main.c -o main
mpirun -np 4 ./main ../data/mat.in ../data/vec.in
rm ./main
