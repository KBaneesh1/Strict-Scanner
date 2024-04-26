#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#pragma pack(1)
typedef struct {
    unsigned short bfType;
    unsigned int   bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int   bfOffBits;
} BITMAPFILEHEADER;

#pragma pack(1)
typedef struct {
    unsigned int   biSize;
    int            biWidth;
    int            biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int   biCompression;
    unsigned int   biSizeImage;
    int            biXPelsPerMeter;
    int            biYPelsPerMeter;
    unsigned int   biClrUsed;
    unsigned int   biClrImportant;
} BITMAPINFOHEADER;

int main(int argc, char *argv[]) {
    if (argc != 7) {
        printf("Usage: %s <image_file_path> <token_file_path> <offx> <offy> <token_dimx> <token_dimy>\n", argv[0]);
        return 1;
    }

    char *image_file_path = argv[1];
    char *token_file_path = argv[2];
    int offx = atoi(argv[3]);
    int offy = atoi(argv[4]);
    int token_dimx = atoi(argv[5]);
    int token_dimy = atoi(argv[6]);

    int image_fd = open(image_file_path, O_RDONLY);
    if (image_fd == -1) {
        printf("Error opening image file!\n");
        return 1;
    }

    struct stat image_stat;
    fstat(image_fd, &image_stat);
    uint8_t *image = mmap(NULL, image_stat.st_size, PROT_READ, MAP_PRIVATE, image_fd, 0);
    if (image == MAP_FAILED) {
        printf("Error mapping image file!\n");
        return 1;
    }

    BITMAPFILEHEADER *bmfh = (BITMAPFILEHEADER *)image;
    BITMAPINFOHEADER *bmih = (BITMAPINFOHEADER *)(image + sizeof(BITMAPFILEHEADER));

    if (bmih->biBitCount != 8 || bmih->biCompression != 0) {
        printf("The image is not an 8-bit grayscale bitmap.\n");
        return 1;
    }

    int image_dimx = bmih->biHeight;
    int image_dimy = bmih->biWidth;
    uint8_t *image_data = image + bmfh->bfOffBits;

    if (offx + token_dimx > image_dimx || offy + token_dimy > image_dimy) {
        printf("The token is out of bounds of the image.\n");
        return 1;
    }

    //create the token matrix
    uint8_t *token = malloc(token_dimx * token_dimy);
    for (int x = 0; x < token_dimx; x++) {
        for (int y = 0; y < token_dimy; y++) {
            token[x * token_dimy + y] = image_data[(offx + x) * image_dimy + offy + y];
        }
    }

    FILE *token_file = fopen(token_file_path, "wb");
    if (token_file == NULL) {
        printf("Error opening token file!\n");
        return 1;
    }

    // copy headers
    BITMAPINFOHEADER* bmih2 = malloc(sizeof(BITMAPINFOHEADER));
    memcpy(bmih2, bmih, sizeof(BITMAPINFOHEADER));

    bmih2->biHeight = token_dimx;
    bmih2->biWidth = token_dimy;

    // copy everything till image_data to token file
    fwrite(image, bmfh->bfOffBits, 1, token_file);
    //seek to start
    fseek(token_file, 0, SEEK_SET);
    // write headers
    fwrite(bmfh, sizeof(BITMAPFILEHEADER), 1, token_file);
    fwrite(bmih2, sizeof(BITMAPINFOHEADER), 1, token_file);
    //seek to end of headers
    fseek(token_file, bmfh->bfOffBits, SEEK_SET);

    // write token matrix
    fwrite(token, token_dimx * token_dimy, 1, token_file);

    fclose(token_file);
    free(token);
    munmap(image, image_stat.st_size);
    close(image_fd);

    return 0;
}
