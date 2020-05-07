mpicc main.c -o main
mpirun -np 4 ./main data.in
rm ./main
