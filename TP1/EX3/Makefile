LIBS= -fopenmp
OBJ= ex3.o
SOURCE= ex3.c
FLAGS= -Wall -g
EXE= ex3

all: build run

run:build

	./${EXE}	
	
build:${SOURCE}

	gcc -c ${SOURCE} ${LIBS} ${FLAGS}
	gcc ${OBJ} ${LIBS} -o ${EXE}
	
clean:

	rm -f ${EXE} ${OBJ}
