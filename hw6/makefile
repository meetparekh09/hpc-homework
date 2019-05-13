MPI = mpicxx
CFLAGS = -std=c++11

TARGETS =  $(basename $(wildcard *.cpp))

all : $(TARGETS)

%:%.cpp
	$(MPI) $(CFLAGS) $< -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
