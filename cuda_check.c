#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

/*
 *  Cuda Device Management
 *  This program talks to the driver which is different
 *  than writting CUDA "Runtime" API code.
 *
 * Outputs some information on CUDA-enabled devices on your computer,
 * including compute capability and current memory usage.
 *
 * On Linux, compile with: nvcc -o cuda_check cuda_check.c -lcuda
 *
 * Authors: Thomas Unterthiner, Jan Schlüter
 *
 * jfdey 2023-05-04 issues
   Cuda 8.2.2.26
   warning: ‘cuCtxDetach’ and 'cuDeviceComputeCapability ' are deprecated
   replace cuCtxDetach with cuCtxDestroy(context);
   replace cuDeviceComputeCapability with cuDeviceGetAttribute [CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 
                                                                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR]

 */

int ConvertSMVer2Cores(int major, int minor)
{
    // Returns the number of CUDA cores per multiprocessor for a given
    // Compute Capability version. There is no way to retrieve that via
    // the API, so it needs to be hard-coded.
    // See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
    switch ((major << 4) + minor) {
        case 0x10: return 8;    // Tesla
        case 0x11: return 8;
        case 0x12: return 8;
        case 0x13: return 8;
        case 0x20: return 32;   // Fermi
        case 0x21: return 48;
        case 0x30: return 192;  // Kepler
        case 0x32: return 192;
        case 0x35: return 192;
        case 0x37: return 192;
        case 0x50: return 128;  // Maxwell
        case 0x52: return 128;
        case 0x53: return 128;
        case 0x60: return 64;   // Pascal
        case 0x61: return 128;
        case 0x62: return 128;
        case 0x70: return 64;   // Volta
        case 0x72: return 64;   // Xavier
        case 0x75: return 64;   // Turing
        default: return 0;
    }
}

int main()
{
    int nGpus;
    int i;
    char name[100];
    int cc_major, cc_minor, cores, cuda_cores, threads_per_core, clockrate;
    size_t freeMem;
    size_t totalMem;

    CUresult result;
    CUdevice device;
    CUcontext context;

    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        printf("cuInit failed with error code %d: %s\n", result, cudaGetErrorString(result));
        return 1;
    }
    result = cuDeviceGetCount(&nGpus);
    if (result != CUDA_SUCCESS) {
        printf("cuDeviceGetCount failed with error code %d: %s\n", result, cudaGetErrorString(result));
        return 1;
    }
    printf("Found %d device(s).\n", nGpus);
    for (i = 0; i < nGpus; i++) {
        cuDeviceGet(&device, i);
        printf("Device: %d\n", i);
        if (cuDeviceGetName(&name[0], sizeof(name), device) == CUDA_SUCCESS) {
            printf("  Name: %s\n", &name[0]);
        }
        if ( cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) == CUDA_SUCCESS &&
             cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) == CUDA_SUCCESS) {
            printf("  Compute Capability: %d.%d\n", cc_major, cc_minor);
        }
        if (cuDeviceGetAttribute(&cores, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) == CUDA_SUCCESS) {
            printf("  Multiprocessors: %d\n", cores);
        }
        if (cuDeviceGetAttribute(&threads_per_core, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) == CUDA_SUCCESS) {
                printf("  Concurrent threads: %d\n", cores*threads_per_core);
        }
        if (cuDeviceGetAttribute(&clockrate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) == CUDA_SUCCESS) {
            printf("  GPU clock: %g MHz\n", clockrate/1000.);
        }
        if (cuDeviceGetAttribute(&clockrate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device) == CUDA_SUCCESS) {
            printf("  Memory clock: %g MHz\n", clockrate/1000.);
        }
        if ( cuCtxCreate(&context, 0, device) == CUDA_SUCCESS) {
            result = cuMemGetInfo(&freeMem, &totalMem);
            if (result == CUDA_SUCCESS ) {
                printf("  Total Memory: %ld MiB\n  Free Memory: %ld MiB\n", totalMem / ( 1024 * 1024 ), freeMem / ( 1024 * 1024 ));
            } else {
                printf("  cMemGetInfo failed with error code %d: %s\n", result, cudaGetErrorString(result));
            }
        }
        cuCtxDestroy(context);
    }
    return 0;
}
