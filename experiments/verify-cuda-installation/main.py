import torch

def main():
    print("PyTorch CUDA Version:", torch.version.cuda)
    print("cuDNN Version:", torch.backends.cudnn.version())

    if torch.cuda.is_available():
        print("CUDA is available. Running on GPU.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Running on CPU.")
        device = torch.device("cpu")

    # Create two tensors and perform a basic operation
    tensor1 = torch.tensor([1.0, 2.0, 3.0]).to(device)
    tensor2 = torch.tensor([4.0, 5.0, 6.0]).to(device)
    result = tensor1 + tensor2

    # Print the result
    print(f"Tensor1: {tensor1}")
    print(f"Tensor2: {tensor2}")
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
