import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()


class PredictionRequest(BaseModel):
    inputs: list[list[float]]


@app.get("/health")
def health():
    return {"status": "healthy"}


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

    return {
        "input_batch": input_batch.tolist(),
        "shape": list(input_batch.shape),
        "dtype": str(input_batch.dtype),
    }

# @app.post("/echo")
# def echo(request: MessageRequest):
#     return {
#         "recived_message": request.message
#     }