EXECS=gauss_mpi
MPICC?=mpicc
LIBS = $(pkg-config --cflags --libs opencv)

all: ${EXECS}

gauss_mpi: gauss_mpi.cpp
	${MPICC} $(LIBS) -o gauss_mpi gauss_mpi.cpp

clean:
	rm -f ${EXECS}
