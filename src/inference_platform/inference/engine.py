"""
This file is responsible for running inference.

Think of it as the "brain" that knows how to:

load a trained model
choose CPU/GPU
receive a batch
run inference
return predictions

"""


from typing import Optional

import torch
from torch import nn

class InferenceEngine:

    """
    Runs PyTorch model inference on CPU,
    Apple MPS,
    or NVIDIA CUDA.
    """


    def __init__(

        self,
        model: nn.Module,
        device: Optional[str] = None,
    ) -> None:

        self.device = self._select_device(device)

        self.model = model.to(self.device)

        self.model.eval()

        print(f"Inference engine initialized on: {self.device}")
    
    @staticmethod
    def _select_device(requested_device: Optional[str]) -> torch.device:
        if requested_device is not None:
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def predict(self, input_batch: torch.Tensor) -> torch.Tensor:
        if input_batch.ndim != 2:
            raise ValueError(
                "Expected input shape [batch_size, features],"
                f"but receivesd {tuple (input_batch.shape)}"

            )

        input_batch =  input_batch.to(self.device)

        with torch.inference_mode():
            output_batch = self.model(input_batch)

        return output_batch.cpu()




    

