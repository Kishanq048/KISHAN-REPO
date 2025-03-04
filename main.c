#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include "kernel.h"

int main(){
	int i, j;
	Kernel k1 = getKernel(3);
	
	pthread_t thread_one;
	pthread_t thread_two;

	Image img1;
	
	img1 = read_image_from_file("text1.txt");

	clock_t begin = clock();

	Image img2 = mul_convolution(img1.data, k1, img1.width, img1.height);

	clock_t end = clock();
	
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("\n");
	for (int i = 0; i < img2.height; i++) {
		for (int j = 0; j < img2.width; j++) {
			//printf("%d\t", img2.data[i][j]);
		}
		//printf("\n");
	}

	/* Creating a new file if it does not exist and write the 
	   Convuluted Image data into them
	*/

	FILE *file = fopen("output_image2.txt", "w");

	for (int i = 0; i < img2.height; i++) {

        for (int j = 0; j < img2.width; j++) {

            fprintf(file, "%d ", img2.data[i][j]);

        }
        fprintf(file, "\n");
    }


	Release(img1.data, img1.height);
	Release(img2.data, img2.height);

	printf("TIME TAKEN IS: %f seconds\n", time_spent);

	return 0;
}