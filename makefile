# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3
LIB = -lm

all: dz1z1 dz1z2 dz1z3 dz1z4 dz1z5

dz1z1: dz1z1.c
	$(CC) $(CC_FLAGS) dz1z1.c -o dz1z1 

dz1z2: dz1z2.c
	$(CC) $(CC_FLAGS) dz1z2.c -o dz1z2 

dz1z3: dz1z3.c
	$(CC) $(CC_FLAGS) dz1z3.c -o dz1z3 $(LIB)

dz1z4: dz1z4.c
	$(CC) $(CC_FLAGS) dz1z4.c -o dz1z4 $(LIB)

dz1z5: dz1z5.c
	$(CC) $(CC_FLAGS) dz1z5.c -o dz1z5 $(LIB)

clean:
	rm -f dz1z1
	rm -f dz1z2
	rm -f dz1z3
	rm -f dz1z4
	rm -f dz1z5