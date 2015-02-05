LIBS= -fopenmp
OBJ= ex2.o
OBJ2= ex2_seq.o
SOURCE= ex2.c
SOURCE2= ex2_seq.c
FLAGS= -Wall -g
EXE= ex2
EXE2=ex2_seq

all: build build2 run

run:build

	./${EXE}	
	
build:${SOURCE}

	gcc -c ${SOURCE} ${LIBS} ${FLAGS}
	gcc ${OBJ} ${LIBS} -o ${EXE}
	
build2:${SOURCE}

	gcc -c ${SOURCE2} ${LIBS} ${FLAGS}
	gcc ${OBJ2} ${LIBS} -o ${EXE2}
	
clean:

	rm -f ${EXE} ${EXE2} ${OBJ} ${OBJ2}
