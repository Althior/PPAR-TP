LIBS= -fopenmp
OBJ= ex5.o
SOURCE= ex5.c
FLAGS= -Wall -g
EXE= ex5

all: build run

run:build

	./${EXE}	
	
build:${SOURCE}

	gcc -c ${SOURCE} ${LIBS} ${FLAGS}
	gcc ${OBJ} ${LIBS} -o ${EXE}
	
clean:

	rm -f ${EXE} ${OBJ}
