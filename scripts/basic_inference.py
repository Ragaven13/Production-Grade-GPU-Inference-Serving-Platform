import torch
from torch import nn


torch.manual_seed(42)

model = nn.Linear(
    in_features=4,
    out_features=3,
)

model.eval()

input_batch = torch.tensor(
    [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 0.0, 3.0],
        [0.5, 1.5, 2.5, 3.5],
    ],
    dtype=torch.float32,
)

with torch.inference_mode():
    output_batch = model(input_batch)
    predicted_classes= torch.argmax(output_batch, dim = 1)

print("Input batch")
print(input_batch)

print("\nInput shape")
print(input_batch.shape)

print("\nOutput batch")
print(output_batch)

print("\nOutput shape")
print(output_batch.shape)


print("\nPredicted classes")
print(predicted_classes)