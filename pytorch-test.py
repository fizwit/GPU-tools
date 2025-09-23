#!/usr/bin/env python3

import torch

COMPUTE_CAPABILITIES = {
    # Format: (compute_version, sm_version)
    "Ampere": [
        (8, 0),  # A100, A40, A30
        (8, 6),  # A10, A16, A2
    ],
    "Hopper": [
        (9, 0),  # H100
    ],
    "Turing": [
        (7, 5),  # T4, Quadro T2000
    ],
    "Volta": [
        (7, 0),  # V100
    ],
    "Pascal": [
        (6, 0),  # P100
        (6, 1),  # P40, P4
    ]
}

def is_compatible_with_device():
    """
    Check if current PyTorch build supports the GPU
    """
    if not torch.cuda.is_available():
        return False
    return True


def get_device_properties():
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" %
          (torch.cuda.device_count(),
          device_id,
          gpu_properties.name,
          gpu_properties.major,
          gpu_properties.minor,
          gpu_properties.total_memory / 1e9))
    compute_cap = gpu_properties.major * 10 + gpu_properties.minor


if __name__ == '__main__':
    if is_compatible_with_device():
        print ("Cuda is available")
        get_device_properties()
    else:
        print('Cuda is not available')

