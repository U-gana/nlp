from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Define the input data structure
class TextInput(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

# Load the model at startup
model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("text_classifier.pkl")

# Define a prediction endpoint
@app.post("/predict")
def predict(input: TextInput):
    # Perform prediction
    prediction = model.predict([input.text])
    return {"text": input.text, "prediction": prediction[0]}