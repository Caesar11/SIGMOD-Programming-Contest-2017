all: main main2 

main.o:main.cpp
	g++ -c main.cpp -g -o main.o -std=c++11 -pthread -Ofast -msse4

main:main.o
	g++ main.o -g -o main -Iinclude -isystem  third-party third-party/lib/libtcmalloc_minimal.a  -pthread -O3 -ltbb -msse4

main2.o:main2.cpp
	g++ -c main2.cpp -g -o main2.o -std=c++11 -pthread -Ofast -msse4

main2:main2.o
	g++ main2.o -g -o main2 -Iinclude -isystem  third-party third-party/lib/libtcmalloc_minimal.a  -pthread -O3 -ltbb -msse4

clean:
	rm -rf main main2 *.o
