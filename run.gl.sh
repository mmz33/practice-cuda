#!/bin/bash

if [ $# != 2 ]; then
  echo "Run CUDA code with OpenGL and Glut dependencies"
  echo "Usage: ./run.gl.sh <file-name> <out-name>"
  exit -1
fi

set -x

LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH
name=$1
out=$2
nvcc $1.cu -o execs/$2 -lGL -lGLU -L build/lib -lglut 
./execs/$2
