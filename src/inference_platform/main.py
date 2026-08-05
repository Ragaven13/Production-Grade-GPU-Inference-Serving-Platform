import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch import nn

from inference_platform.inference.engine import InferenceEngine


app = FastAPI()


class PredictionRequest(BaseModel):
    inputs: list[list[float]]


def create_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )


torch.manual_seed(42)

model = create_model()
engine = InferenceEngine(model=model)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(engine.device),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if len(request.inputs) == 0:
        raise HTTPException(
            status_code=400,
            detail="The input batch cannot be empty.",
        )

    for sample_index, sample in enumerate(request.inputs):
        if len(sample) != 4:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Sample {sample_index} must contain exactly "
                    f"4 features, but received {len(sample)}."
                ),
            )

    input_batch = torch.tensor(
        request.inputs,
        dtype=torch.float32,
    )

    output_batch = engine.predict(input_batch)

    predicted_classes = torch.argmax(
        output_batch,
        dim=1,
    )

    return {
        "predictions": predicted_classes.tolist(),
        "logits": output_batch.tolist(),
        "batch_size": input_batch.shape[0],
        "device": str(engine.device),
    }