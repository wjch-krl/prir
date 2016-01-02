EXECS=gauss_mpi
MPICC?=mpicc

all: ${EXECS}

gauss_mpi: 
	${MPICC} gaussianBlur.cpp -o gauss_mpi -lopencv_highgui -lopencv_core -I/opt/opencv/include -L/opt/opencv/lib

clean:
	rm -f ${EXECS}
