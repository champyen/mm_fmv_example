CC=clang -O3
#CC=gcc -O3

all: mm mm_vec mm_omp mm_vec_omp

mm: mm.c
	$(CC) -o mm mm.c

mm_vec: mm.c
	$(CC)  -DENABLE_VEC -o mm_vec mm.c

mm_omp: mm.c
	$(CC) -o mm_omp mm.c -fopenmp -pthread

mm_vec_omp: mm.c
	$(CC) -DENABLE_VEC -o mm_vec_omp mm.c -fopenmp -pthread

clean:
	rm -f mm mm_vec mm_omp mm_vec_omp