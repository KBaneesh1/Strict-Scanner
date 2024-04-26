
bmp_strict_scanner: bmp_strict_scanner.c
	gcc -o bmp_strict_scanner bmp_strict_scanner.c

strict_scanner: strict_scanner.c
	nvcc -o strict_scanner strict_scanner.cu

run: bmp_strict_scanner
	./bmp_strict_scanner lena_gray.bmp scan.bmp
