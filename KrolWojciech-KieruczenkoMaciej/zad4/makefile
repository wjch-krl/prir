EXECS=gauss_mpi
MPICC?=mpicc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all: ${EXECS}

gauss_mpi: gauss_mpi.cpp
	${MPICC} $(CFLAGS) $(LIBS) -o gauss_mpi gauss_mpi.cpp

clean:
	rm -f ${EXECS}
