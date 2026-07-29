import time 

import torch

from torch import nn

from basic_inference import input_batch

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(1024,2048),
    nn.ReLU(),
    nn.Linear(2048, 1024),
    nn.Linear(1024,10)
)

model.eval()

batch_size = 100

number_of_features = 1024

input_batch = torch.randn(
    batch_size,
    number_of_features,
    dtype=torch.float32,
)

with torch.inference_mode():

    _ = model(input_batch)

    individual_start = time.perf_counter()

    individual_outputs = []

    for sample in input_batch:
        sample = sample.unsqueeze(0)
        output = model(sample)
        individual_outputs.append(output)
        


individual_end= time.perf_counter()

batch_start = time.perf_counter()

batch_output = model(input_batch)

batch_end = time.perf_counter()

individual_time = individual_end - individual_start
batch_time = batch_end  - batch_start

individual_throughput = batch_size / individual_time

batch_throughput = batch_size / batch_time

print(f"Number of samples: {batch_size}")

print("\nIndividual inference")
print(f"Total time: {individual_time:.6f} seconds")
print(f"Throughput: {individual_throughput:.2f} samples/second")

print("\nBatched inference")
print(f"Total time: {batch_time:.6f} seconds")
print(f"Throughput: {batch_throughput:.2f} samples/second")

print("\nOutput shapes")
print(f"Individual outputs: {len(individual_outputs)} tensors")
print(f"Each individual output: {individual_outputs[0].shape}")
print(f"Batched output: {batch_output.shape}")