#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <pwd.h>
#include <nvml.h>

/* build me:
 *    nvcc -lcuda -lnvidia-ml  nvps.c -o nvps
 *
 *   build with this module: cuDNN/8.0.4.30-CUDA-11.1.1
 *    or  cuDNN/9.5.1.17-CUDA-12.6.2
 *   fails with 11.7
 *   Nvidia Device driver version 470.141.03 only has "_v2" NVML library calls
 *
 */

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
    char username[256];
    nvmlPstates_t pState;
    nvmlProcessInfo_t infos[32];

    printf("    {\n      \"device\": %d,\n", device_num);
    /* get MIG device ID */
    if ((result = nvmlDeviceGetHandleByIndex(device_num, &device)) !=  NVML_SUCCESS) {
        fprintf(stderr, "error: nvmlDeviceGetHandleByIndex: %s\n", nvmlErrorString(result));
        exit(EXIT_FAILURE);
    }
    if ((result = nvmlDeviceGetName( device, name, NVML_DEVICE_NAME_BUFFER_SIZE)) != NVML_SUCCESS) {
        fprintf(stderr, "error: Get Driver Model: %s\n", nvmlErrorString(result));
        exit(EXIT_FAILURE);
    }
    printf("      \"Device_Name\": \"%s\",\n", name);

    if ((result = nvmlDeviceGetMemoryInfo(device, &memory))  != NVML_SUCCESS) {
        fprintf(stderr, "error: Get Compute Running Processes error: %s\n", nvmlErrorString(result));
        exit(EXIT_FAILURE);
    }
    printf("      \"Memory_Total\": %lld,\n", memory.total);
    printf("      \"Memory_used\": %lld", memory.used);

    infoCount = 32;
    if ((result = nvmlDeviceGetComputeRunningProcesses(device, &infoCount, infos)) != NVML_SUCCESS) {
        fprintf(stderr, "error: Get Compute Running Processes error: %s\n", nvmlErrorString(result));
        exit(EXIT_FAILURE);
    }

    if ( infoCount > 0  )
        printf(",\n      \"PIDS\": [\n");
    else
        printf("\n    }");

    for (i=0; i<infoCount; i++) {
        if ( i > 0 ) printf(",\n");
        if ((result = nvmlSystemGetProcessName(infos[i].pid, pname, 256)) != NVML_SUCCESS) {
            fprintf(stderr, "nvmlSystemGetProcessName: %s\n", nvmlErrorString(result));
            exit(EXIT_FAILURE);
        }
        if (get_uid_from_pid_stat(infos[i].pid, &uid) != -1) {
            get_username_from_uid(uid, username, sizeof(username));
            printf("        [%2d, %u, \"%s\", \"%s\"]", device_num, uid, username, pname);
        }
        else
            printf("        [%2d, -1, \"NA\", \"%s\"]", device_num, pname);
    }
    if ( infoCount > 0  )
        printf("\n      ]\n    }");
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

    printf("{\n");
    if ((result = gethostname(hname, 128)) ) {
        fprintf(stderr, "gethostname: %s\n", strerror(result));
        return 1;
    }
    printf("  \"host\": \"%s\",\n", hname);

    if ((result = nvmlSystemGetDriverVersion(version, (unsigned int)64)) != NVML_SUCCESS) {
        fprintf(stderr, "nvmlSystemGetDriverVersion: %s\n", nvmlErrorString(result));
        return 1;
    }
    printf("  \"Driver_Version\": \"%s\",\n", version);

    if ((result = nvmlSystemGetCudaDriverVersion(&cudaDriverVersion)) != NVML_SUCCESS) {
        fprintf(stderr, "System Get Cuda Version: %s\n", nvmlErrorString(result));
        return 1;
    }
    printf("  \"CUDA_Version\": \"%d.%d\",\n", NVML_CUDA_DRIVER_VERSION_MAJOR(cudaDriverVersion),
                                    NVML_CUDA_DRIVER_VERSION_MINOR(cudaDriverVersion));

    if ((result = nvmlSystemGetNVMLVersion(version, (unsigned int)64)) != NVML_SUCCESS) {
        fprintf(stderr, "nvmlSystemGetNVMLVersion: %s\n", nvmlErrorString(result));
        return 1;
    }
    printf("  \"NVML_library_version\": \"%s\",\n", version);

    if (deviceCount > 0)
       printf("  \"GPU\": [\n");
        for(i=0; i<deviceCount; i++) {
           device_info(i);
           if ( i < (deviceCount - 1))
               printf(",\n");
           else
               printf("\n");
       }
       printf("   ]\n");
   printf("}\n");
}

