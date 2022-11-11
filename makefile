# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3
LIB = -lm
OBJ = main.o \
	dfill.o \
	domove.o \
	dscal.o \
	fcc.o \
	forces.o \
	mkekin.o \
	mxwell.o \
	prnout.o \
	velavg.o

all: prime feyman md

prime: 
	$(CC) $(CC_FLAGS) prime.c -o prime 

feyman: 
	$(CC) $(CC_FLAGS) feyman.c -o feyman $(LIB)

md:	$(OBJ)
	$(CC) $(CC_FLAGS) -o $@ $(OBJ) $(LIB)

.c.o:
	$(CC) $(CC_FLAGS) -c $<

clean:
	rm -f prime
	rm -f feyman
	rm *.o md