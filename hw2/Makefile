make: MMult1 val_test01_solved val_test02_solved omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 gs2D-omp jacobi2D-omp

MMult1: MMult1.cpp
	g++ -march=native -O3 -fopenmp -std=c++11 MMult1.cpp -o MMult1

val_test01_solved: val_test01_solved.cpp
	g++ val_test01_solved.cpp -o val_test01_solved

val_test02_solved: val_test02_solved.cpp
	g++ val_test02_solved.cpp -o val_test02_solved

omp_solved2: omp_solved2.c
	g++ -fopenmp omp_solved2.c -o omp_solved2

omp_solved3: omp_solved3.c
	g++ -fopenmp omp_solved3.c -o omp_solved3

omp_solved4: omp_solved4.c
	g++ -fopenmp omp_solved4.c -o omp_solved4

omp_solved5: omp_solved5.c
	g++ -fopenmp omp_solved5.c -o omp_solved5

omp_solved6: omp_solved6.c
	g++ -fopenmp omp_solved6.c -o omp_solved6

jacobi2D-omp: jacobi2D-omp.cpp
	g++ -fopenmp -std=c++11 -O3 -march=native jacobi2D-omp.cpp -o jacobi2D-omp

gs2D-omp: gs2D-omp.cpp
	g++ -fopenmp -std=c++11 -O3 -march=native gs2D-omp.cpp -o gs2D-omp

clean:
	rm MMult1 val_test01_solved val_test02_solved omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 gs2D-omp jacobi2D-omp
