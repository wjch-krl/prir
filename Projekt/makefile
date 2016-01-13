EXECS=proj
MPICC?=mpic++

all: ${EXECS}

proj: 
	${MPICC} main.cpp -std=c++11 -lopencv_highgui -lopencv_imgcodecs -lopencv_core -I/usr/local/include -L/usr/local/lib -o proj

clean:
	rm -f ${EXECS}
