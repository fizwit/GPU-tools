#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <pwd.h>
#include <nvml.h>
#include <ctype.h>

/*
   john dey 2025.08.24

   nvidia-info - sytem admin tools for displaying nvidia GPU device info

   Addtions from nvidia-smi
   output the the UID and uname, but drop the process id (PID)
   oupput memory in human readable form, 14 digit integers are hard to read
   comma's would also help.
   output the the device ID and device serial. The serial should
   match the engraved serial on the physical GPU case.
   The device ID is the location on PCI bus, but I don' know how to
   use this info to find the phicical device.

   Build Me:
   Use the Nvidia nvcc compiler

      nvidia-info -lcuda -lnvidia-ml  nvidia-info.c -o nvidia-info

   Use this LMOD module  cuDNN/8.0.4.30-CUDA-11.1.1
      or  cuDNN/9.5.1.17-CUDA-12.6.2
     fails with 11.7
     Nvidia Device driver version 470.141.03 only has "_v2" NVML library calls

 */

/* convert long long int of bytes into GB */
void
human( long long int s, char *h )
{
    char *unit[] = {"B", "KB", "MB", "GB", "TB"};
    double val;
    int ind =0;

    val = (double)s;
    while ( val > 1024.0 ) {
       val = val/1024.0;
       ind++;
    }
    sprintf(h, "%.2f%s", val, unit[ind] );
}

int 
get_username_from_uid(uid_t uid, char *username, size_t size) {
    struct passwd *pw;

    pw = getpwuid(uid);
    if (pw == NULL) {
        return -1; // UID not found
    }
    snprintf(username, size, "%s", pw->pw_name);
    return 0;
}

int
get_uid_from_pid_stat(pid_t pid, uid_t *uid)
{
    char path[256];
    struct stat st;
    struct passwd *pw;

    snprintf(path, sizeof(path), "/proc/%d", pid);
    if (stat(path, &st) == 0) {
        *uid = st.st_uid;
        return 0;
    }
    return -1; // Process doesn't exist or no permission
}

void
device_info(int device_num)
{
    int i;
    nvmlReturn_t result;
    nvmlMemory_t memory;
    nvmlDevice_t device;
    char pname[256], name[NVML_DEVICE_NAME_BUFFER_SIZE];
    int major, minor;
    unsigned int infoCount, temp;
    uid_t uid;
    char username[256], serial[256], human_val[256];
    nvmlPstates_t pState;
    nvmlProcessInfo_t infos[32];

    printf("  GPU: %d\n", device_num);
    /* get MIG device ID */
    if ((result = nvmlDeviceGetHandleByIndex(device_num, &device)) !=  NVML_SUCCESS) {
        fprintf(stderr, "error: nvmlDeviceGetHandleByIndex: %s\n", nvmlErrorString(result));
        exit(EXIT_FAILURE);
    }
    printf("   Device ID: %ld\n", device);
    if ((result = nvmlDeviceGetName( device, name, NVML_DEVICE_NAME_BUFFER_SIZE)) != NVML_SUCCESS) {
        fprintf(stderr, "error: Get Driver Model: %s\n", nvmlErrorString(result));
        exit(EXIT_FAILURE);
    }
    printf("   Device Name: %s\n", name);

    if ((result = nvmlDeviceGetSerial ( device, serial, (unsigned int)245 )) != NVML_SUCCESS) {
        fprintf(stderr, "error: Device Get Serial: %s\n", nvmlErrorString(result));
        exit(EXIT_FAILURE);
    }
    printf("   Serial: %s\n", serial);

    if ((result = nvmlDeviceGetMemoryInfo(device, &memory))  != NVML_SUCCESS) {
        fprintf(stderr, "error: Get Compute Running Processes error: %s\n", nvmlErrorString(result));
        exit(EXIT_FAILURE);
    }
    human((long long int)memory.total, human_val);
    printf("   Memory_Total: %16lld (%s)\n", memory.total, human_val);
    human((long long int)memory.used, human_val);
    printf("   Memory_used:  %16lld (%s)\n", memory.used, human_val);

    infoCount = 32;
    if ((result = nvmlDeviceGetComputeRunningProcesses(device, &infoCount, infos)) != NVML_SUCCESS) {
        fprintf(stderr, "error: Get Compute Running Processes error: %s\n", nvmlErrorString(result));
        exit(EXIT_FAILURE);
    }

    if ( infoCount > 0  )
        printf("   PIDS: [\n");

    for (i=0; i<infoCount; i++) {
        if ( i > 0 ) printf(",\n");
        if ((result = nvmlSystemGetProcessName(infos[i].pid, pname, 256)) != NVML_SUCCESS) {
            fprintf(stderr, "nvmlSystemGetProcessName: %s\n", nvmlErrorString(result));
            exit(EXIT_FAILURE);
        }
        if (get_uid_from_pid_stat(infos[i].pid, &uid) != -1) {
            get_username_from_uid(uid, username, sizeof(username));
            printf("        [%2d, %u, \"%s\", \"%s\"]\n", device_num, uid, username, pname);
        }
        else
            printf("        [%2d, -1, \"NA\", \"%s\"]\n", device_num, pname);
    }
    return;
}

int
main(int argc, char* argv[])
{
    int i;

    unsigned int deviceCount;
    nvmlReturn_t result;
    char version[64];
    char hname[128];
    int cudaDriverVersion;

    if ((result = nvmlInit()) != NVML_SUCCESS) {
        fprintf(stderr, "Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return 1;
    }

    if ((result = nvmlDeviceGetCount(&deviceCount)) != NVML_SUCCESS) {
        fprintf(stderr, "no GPU found\n");
        return 1;
    }

    if ((result = nvmlSystemGetDriverVersion(version, (unsigned int)64)) != NVML_SUCCESS) {
        fprintf(stderr, "nvmlSystemGetDriverVersion: %s\n", nvmlErrorString(result));
        return 1;
    }
    printf("  Driver_Version: %s,\n", version);

    if ((result = nvmlSystemGetCudaDriverVersion(&cudaDriverVersion)) != NVML_SUCCESS) {
        fprintf(stderr, "System Get Cuda Version: %s\n", nvmlErrorString(result));
        return 1;
    }
    printf("  CUDA_Version: %d.%d\n", NVML_CUDA_DRIVER_VERSION_MAJOR(cudaDriverVersion),
                                    NVML_CUDA_DRIVER_VERSION_MINOR(cudaDriverVersion));

    if ((result = nvmlSystemGetNVMLVersion(version, (unsigned int)64)) != NVML_SUCCESS) {
        fprintf(stderr, "nvmlSystemGetNVMLVersion: %s\n", nvmlErrorString(result));
        return 1;
    }
    printf("  NVML_library_version: %s\n", version);

    if (deviceCount > 0)
        for(i=0; i<deviceCount; i++)
           device_info(i);
}

