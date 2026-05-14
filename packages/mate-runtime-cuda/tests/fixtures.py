# nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader,nounits

SMI_RTX4090 = "NVIDIA GeForce RTX 4090, 24564, 550.54.15, 12.4"
SMI_RTX3080 = "NVIDIA GeForce RTX 3080, 10240, 535.183.01, 12.2"
SMI_A100    = "NVIDIA A100-SXM4-80GB, 81920, 520.61.05, 11.8"
SMI_UNKNOWN = "NVIDIA SomeUnknownGPU, 8192, 550.54.15, 12.4"
SMI_EMPTY   = ""

# nvidia-smi -L
SMI_LIST_RTX4090 = "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-abc123)\n"
SMI_LIST_EMPTY   = ""
