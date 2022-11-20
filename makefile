# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3

%: %.c
	$(CC) $(CC_FLAGS) -o $(@) $(<) -lm

clean:
	rm -f *.o
