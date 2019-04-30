MPI = mpicc
CFLAGS = -std=c99

TARGETS =  $(basename $(wildcard *.c))

all : $(TARGETS)

%:%.c
	$(MPI) $(CFLAGS) $< -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
