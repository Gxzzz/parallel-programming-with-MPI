mpicc main.c -o main -lm
mpirun -np 4 ./main $1
rm ./main
