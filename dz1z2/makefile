# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3

%: %.c
	$(CC) $(CC_FLAGS) -o $(@) $(<)

clean:
	rm -f *.o
