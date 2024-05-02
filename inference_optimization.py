import argparse
import torch
import torchvision
from torchvision.models import resnet18, densenet121
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import time
import matplotlib.pyplot as plt

def get_test_loader():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    return test_loader

def inference(model, dataloader):
    start = time.time()
    for images, labels in dataloader:
        outputs = model(images)
    end = time.time()
    return end - start

def run_inference(model, test_loader, optimize=False):
    inference_times = []

    # Inference without optimization
    inference_time_without_optimization = inference(model, test_loader)
    print("Inference time without optimization:", inference_time_without_optimization)
    inference_times.append(inference_time_without_optimization)

    if optimize:
        # Quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        inference_time_with_low_res_quant = inference(quantized_model, test_loader)
        print("Inference time with low-resolution quantification:", inference_time_with_low_res_quant)
        inference_times.append(inference_time_with_low_res_quant)

        # TorchScript
        scripted_model = torch.jit.script(model)
        inference_time_with_torch = inference(scripted_model, test_loader)
        print("Inference time with TorchScript:", inference_time_with_torch)
        inference_times.append(inference_time_with_torch)

    return inference_times

def plot_comparison(inference_times, model_name):
    optimization_techniques = ['No Optimization', 'Quantization', 'TorchScript']
    plt.bar(optimization_techniques[:len(inference_times)], inference_times)
    plt.xlabel('Optimization Technique')
    plt.ylabel('Inference Time (seconds)')
    plt.title(f'Inference Time Comparison for {model_name}')
    plt.show()

def main(args):
    test_loader = get_test_loader()

    # ResNet
    if args.model == 'resnet':
        model = resnet18(pretrained=True)
    # DenseNet
    elif args.model == 'densenet':
        model = densenet121(pretrained=True)
    else:
        raise ValueError("Invalid model name. Please choose 'resnet' or 'densenet'.")

    model.eval()
    inference_times = run_inference(model, test_loader, optimize=(args.optimization == 'optimize'))
    plot_comparison(inference_times, args.model.capitalize())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference Time Comparison for ResNet and DenseNet models')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'densenet'], help='Choose the model to evaluate (default: resnet)')
    parser.add_argument('--optimization', type=str, default='none', choices=['none', 'optimize'], help='Choose whether to perform optimizations (default: none)')
    args = parser.parse_args()
    main(args)