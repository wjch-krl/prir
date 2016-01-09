EXECS=gauss_mpi
MPICC?=mpic++

all: ${EXECS}

gauss_mpi: 
	${MPICC} gaussianBlur.cpp -std=c++11 -lopencv_highgui -lopencv_imgcodecs -lopencv_core -I/usr/local/include -L/usr/local/lib -o gaus_mpi

clean:
	rm -f ${EXECS}
