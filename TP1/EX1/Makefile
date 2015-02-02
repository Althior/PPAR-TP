LIBS= -fopenmp
OBJ= ex1.o
SOURCE= ex1.c
FLAGS= -Wall -g
EXE= ex1

all: build run

run:build

	./${EXE}	
	
build:${SOURCE}

	gcc -c ${SOURCE} ${LIBS} ${FLAGS}
	gcc ${OBJ} ${LIBS} -o ${EXE}
	
clean:

	rm -f ${EXE} ${OBJ}
