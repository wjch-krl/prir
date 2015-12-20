EXECS=gauss_mpi
MPICC?=mpicc

all: ${EXECS}

gauss_mpi: gauss_mpi.cpp
	${MPICC}  `pkg-config --cflags --libs opencv` -o gauss_mpi gauss_mpi.cpp

clean:
	rm -f ${EXECS}
