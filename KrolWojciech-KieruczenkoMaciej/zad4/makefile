EXECS=gauss_mpi
MPICC?=mpic++

all: ${EXECS}

gauss_mpi: 
	${MPICC} gaussianBlur.cpp -std=c++11 -lopencv_highgui -lopencv_core -I/opt/opencv/include -L/opt/opencv/lib -o gaus_mpi

clean:
	rm -f ${EXECS}
