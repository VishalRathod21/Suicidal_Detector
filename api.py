from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import uvicorn

app = FastAPI(
    title="Suicidal Post Detection API",
    description="API for detecting potential suicidal content in text",
    version="1.0.0"
)

# Load model and tokenizer
try:
    tokenizer = pickle.load(open('models/tokenizer.pkl', 'rb'))
    model = load_model("models/model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: float
    is_suicidal: bool
    confidence: float

@app.get("/")
async def root():
    return {"message": "Welcome to Suicidal Post Detection API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    try:
        # Preprocess the text
        text_sequence = tokenizer.texts_to_sequences([input_data.text])
        padded_sequence = pad_sequences(text_sequence, maxlen=50)
        
        # Make prediction
        prediction = model.predict(padded_sequence)[0][0]
        
        # Prepare response
        response = {
            "prediction": float(prediction),
            "is_suicidal": bool(prediction > 0.5),
            "confidence": float(prediction * 100 if prediction > 0.5 else (1 - prediction) * 100)
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 