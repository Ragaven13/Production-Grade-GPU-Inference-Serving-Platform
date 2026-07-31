from email import message
from fastapi import FastAPI
from pydantic import BaseModel



app = FastAPI()

class PredictionRequest(BaseModel):
    input: list[list[float]]



@app.get("/health")
def health():
    return {"status":"healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    return {
        "recived_inputs": request.inputs
    }



# @app.post("/echo")
# def echo(request: MessageRequest):
#     return {
#         "recived_message": request.message
#     }