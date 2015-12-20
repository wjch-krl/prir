EXECS=gauss_mpi
MPICC?=mpicc

all: ${EXECS}

gauss_mpi: gauss_mpi.cpp
	${MPICC} -o gauss_mpi gauss_mpi.c

clean:
	rm -f ${EXECS}
