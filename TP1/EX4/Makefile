LIBS= -fopenmp
OBJ= ex4.o
OBJ2= ex4_seq.o
SOURCE= ex4.c ex4_seq.c
FLAGS= -Wall -g
EXE= ex4
EXE2= ex4_seq

all: build run

run:build

	./${EXE}	
	
build:${SOURCE}

	gcc -c ${SOURCE} ${LIBS} ${FLAGS}
	gcc ${OBJ} ${LIBS} -o ${EXE}
	gcc ${OBJ2} ${LIBS} -o ${EXE2}
	
clean:

	rm -f ${EXE} ${EXE2} ${OBJ} ${OBJ2}
