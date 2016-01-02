EXECS=gauss_mpi
MPICC?=mpicc

all: ${EXECS}

gauss_mpi: gauss_mpi
	${MPICC} -o gauss_mpi -lopencv_highgui -lopencv_core -I/opt/opencv/include -L/opt/opencv/lib

clean:
	rm -f ${EXECS}
