# Build Notes

For sites that support LMOD modules. Load the Nvida compiler cuDNN
```
ml cuDNN/8.2.2.26-CUDA-11.4.1
 * build cuda_check
 ```
nvcc nvps.c -o nvps -lcuda -lnvidia-ml
```
