#!/bin/bash

export OMP_NUM_THREADS=16

CC="/usr/local/bin/mpicxx"
CFLAGS="-Xpreprocessor -fopenmp -std=c++14"
IFLAGS=""
LFLAGS="-lomp -lhdf5"

# Build executable
compile () {
  rm -f $1
  ${CC} ${CFLAGS} ${IFLAGS} -c $1.cc -o $1.o
  ${CC} ${CFLAGS} $1.o ${LFLAGS} -o $1
}

# Run executable
run () {
  compile $1
  ./$1 ../Thesan-1 output 54
}

#compile thesan-mfp
run thesan-mfp


