# Build Notes

For sites that support LMOD modules.
```
ml cuDNN/8.2.2.26-CUDA-11.4.1
 * build cuda_check
 ```
nvcc -o cuda_check cuda_check.c -lcuda
 ```
