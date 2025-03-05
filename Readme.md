# Image processing using OpenMP



# How to run

Images obtained after running the project are:

[Darker image](./assets/dark.jpg)   
[Lighter image](./assets/light.jpg)   


The Kernels used here, for demonstration purposes, respectively, are:
1. For the darker image:

    ```python3
    0 -1  0
    -1  5 -1
    0 -1  0
    ```

2. For the lighter image:

    ```python3
    5 2 1
    0 1 0
    -2 0 -2
    ```