LIBS= -fopenmp
OBJ= ex3.o
OBJ2= ex3_seq.o
SOURCE= ex3.c ex3_seq.c
FLAGS= -Wall -g
EXE= ex3
EXE2= ex3_seq

all: build run

run:build

	./${EXE}	
	
build:${SOURCE}

	gcc -c ${SOURCE} ${LIBS} ${FLAGS}
	gcc ${OBJ} ${LIBS} -o ${EXE}
	gcc ${OBJ2} ${LIBS} -o ${EXE2}
	
clean:

	rm -f ${EXE} ${EXE2} ${OBJ} ${OBJ2}
