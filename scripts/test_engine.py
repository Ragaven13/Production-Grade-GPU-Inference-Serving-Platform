import torch
from torch import nn

from inference_platform.inference.engine import InferenceEngine


def create_model() -> nn.Module:
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )

    return model


def main() -> None: 
    torch.manual_seed(42)

    model = create_model()
    engine = InferenceEngine(model=model)

    input_batch = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 0.0, 3.0],
            [0.5, 1.5, 2.5, 3.5],
        ],
        dtype=torch.float32,
    )

    output_batch = engine.predict(input_batch)

    predicted_classes = torch.argmax(output_batch, dim=1)

    print("\nInput shape:")
    print(input_batch.shape)

    print("\nOutput batch:")
    print(output_batch)

    print("\nOutput shape:")
    print(output_batch.shape)

    print("\nPredicted classes:")
    print(predicted_classes)


if __name__ == "__main__":
    main()