MPI = mpicc

TARGETS =  $(basename $(wildcard *.c))

all : $(TARGETS)

%:%.c
	$(MPI) $< -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
