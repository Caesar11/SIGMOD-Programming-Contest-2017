#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# echo ${DIR}
# pip install --user ${DIR}/src/packages/*
cd src/third-party/gperftools-gperftools-2.5
sh autogen.sh
./configure --prefix=${DIR}/src/third-party --disable-cpu-profiler --disable-heap-profiler --disable-heap-checker --disable-debugalloc --enable-minimal 
make
make install
cd ../..
make main

