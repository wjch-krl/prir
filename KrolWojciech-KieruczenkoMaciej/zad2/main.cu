#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include "matrix.cu"

int main(int argc, char *argv[])
{
	Matrix *m;
	std::string path;
	if(argc != 2)
	{
		/* Print usage */
		std::cout<< "USSAGE:\n "<< argv[0] <<" FILE_PATH\n";
		return EXIT_FAILURE;
	}
	path = argv[1];

	m = new Matrix(path,10,50);
	/*
	Time mesurment using std::chrono.
	I personally dont think that in this case using cuda events will be better,
	becouse we are mesuring time of entire process which is part only accelated using GPU 
	(Some important calculation are made on CPU)
	*/
	auto t1 = std::chrono::high_resolution_clock::now();
	/* Calculate determinant, next print value and computation time */
	long double det = m->determinant();
	auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " <<  std::chrono::duration_cast<std::chrono::milliseconds>(t2 
		- t1).count() << " ms\n";
    std::cout << "Determinant value: " << det << "\n";
	/* Memory cleanup */
	delete m;
	return 0;
}
