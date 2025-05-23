from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn

# Load model and vectorizer
model = joblib.load("Url_phishing.pkl")
vectorizer = joblib.load("url_VECtorizer.pkl")

app = FastAPI()

# Define input data schema
class UrlInput(BaseModel):
    url: str

@app.post("/predict")
def predict_phishing(data: UrlInput):
    url = data.url
    try:
        # Vectorize the input URL (assuming your model only needs vectorized URL)
        url_vector = vectorizer.transform([url])
        
        # Predict
        pred = model.predict(url_vector)[0]
        # You can map 0/1 to labels if you want
        label = "phishing" if pred == 1 else "legit"
        
        return {"prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: for local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
