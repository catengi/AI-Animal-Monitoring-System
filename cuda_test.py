import torch
# Check if GPU is available
print("CUDA Available:", torch.cuda.is_available())

# Get GPU details
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))