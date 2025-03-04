#include "kernel.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

Kernel getKernel(int size) {
	int i, j;
	Kernel k1;
	
	// dynamically allocating outer array
	float **array = (float **)(malloc(sizeof(float *) * size)); 
	
	for (i = 0; i < size; i++) {
		// looping through array and allocating memory
		array[i] = (float *)(malloc(sizeof(float) * size)); 
		
	}
	
	printf("Enter the kernel values\n");
	// Flushing the buffer output
	fflush(stdout); 
	
	// Reading values of the kernel from the user
	for (i = 0; i < size; i++) { 	
		for (j = 0; j < size; j++) {
			scanf("%f", &array[i][j]);
		}
	}
	
	k1.data = array;
	k1.size = size;
	
	return k1;
}

void readKernel(Kernel k) {
	int i;
	int j;
	
	printf("Kernel\n");
	fflush(stdout);
	
	// Displaying values of the kernel
	for (i = 0; i < k.size; i++) { 	
		for (j = 0; j < k.size; j++) {
			printf("%f\t", k.data[i][j]);
		}
		printf("\n");
	}
}

Image convolution(int **input_image, Kernel k, int width, int height) {
    int i, j, m, n;
    int sum;

    // Create an output Image structure
    Image img;
    img.width = width;
    img.height = height;

    // Allocate memory for the output image data
    img.data = (int **)(malloc(sizeof(int *) * height));
    for (i = 0; i < height; i++) {
        img.data[i] = (int *)(malloc(sizeof(int) * width));
    }

    // Perform convolution
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            sum = 0;

            // Apply the kernel to the current position
            for (m = 0; m < k.size; m++) {
                for (n = 0; n < k.size; n++) {
                    int x = i + m - k.size / 2; // Adjust for the kernel center
                    int y = j + n - k.size / 2;

                    // Check for boundary conditions
                    if (x >= 0 && x < height && y >= 0 && y < width) {
                        sum += input_image[x][y] * k.data[m][n];
                    }
                }
            }

            img.data[i][j] = sum;
        }
    }

    return img;
}

void* thread_convolution(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int **input_image = data->input_image;
    int **output_image = data->output_image;
    Kernel k = data->k;
    int width = data->width;
    int height = data->height;
    int start_row = data->start_row;
    int end_row = data->end_row;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < width; j++) {
            int sum = 0;
            for (int m = 0; m < k.size; m++) {
                for (int n = 0; n < k.size; n++) {
                    int x = i + m - k.size / 2;
                    int y = j + n - k.size / 2;
                    if (x >= 0 && x < height && y >= 0 && y < width) {
                        sum += input_image[x][y] * k.data[m][n];
                    }
                }
            }
            output_image[i][j] = sum;
        }
    }

    return NULL;
}

Image* addPadding(Image* input_image, int padding) {
    int new_width = input_image->width + 2 * padding;
    int new_height = input_image->height + 2 * padding;
    
    // Create new padded image
    Image* padded_image = (Image*)malloc(sizeof(Image));
    padded_image->width = new_width;
    padded_image->height = new_height;
    
    // Allocate memory for padded image data
    padded_image->data = (int**)malloc(new_height * sizeof(int*));
    for (int i = 0; i < new_height; i++) {
        padded_image->data[i] = (int*)malloc(new_width * sizeof(int));
        
        // Initialize all elements to 0 (zero padding)
        for (int j = 0; j < new_width; j++) {
            padded_image->data[i][j] = 0;
        }
    }
    
    // Copy the original image into the center of the padded image
    for (int i = 0; i < input_image->height; i++) {
        for (int j = 0; j < input_image->width; j++) {
            padded_image->data[i + padding][j + padding] = input_image->data[i][j];
        }
    }
    
    return padded_image;
}

void Release(int **array, int rows) {
	int i;
	
	for (i = 0; i < rows; i++) {
		free(array[i]);
	}
	
	free(array);
}

Image read_image_from_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    int width, height;
    fscanf(fp, "%d %d", &height, &width);

    Image image;
    image.width = width;
    image.height = height;

    image.data = (int **)malloc(height * sizeof(int *));
    if (image.data == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    for (int i = 0; i < height; i++) {
        image.data[i] = (int *)malloc(width * sizeof(int));
        if (image.data[i] == NULL) {
            fprintf(stderr, "Memory allocation error\n");
            exit(1);
        }
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (fscanf(fp, "%d", &image.data[i][j]) != 1) {
                fprintf(stderr, "Error reading data from file\n");
                exit(1);
            }
        }
    }

    fclose(fp);
    return image;
}

Image mul_convolution(int **input_image, Kernel k, int width, int height) {
    int num_threads = 7; // Number of threads
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    int rows_per_thread = height / num_threads;

    Image img;
    img.width = width;
    img.height = height;
    img.data = (int **)(malloc(sizeof(int *) * height));
    for (int i = 0; i < height; i++) {
        img.data[i] = (int *)(malloc(sizeof(int) * width));
    }

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input_image = input_image;
        thread_data[i].output_image = img.data;
        thread_data[i].k = k;
        thread_data[i].width = width;
        thread_data[i].height = height;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? height : (i + 1) * rows_per_thread;
        pthread_create(&threads[i], NULL, thread_convolution, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
       //pthread_join(threads[i], NULL);
    }

    return img;
}