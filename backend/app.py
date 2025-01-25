from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware

# Initialize the app
app = FastAPI()

# Allow all origins (or specify allowed origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your React app's URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow GET and POST methods
    allow_headers=["*"],  # Allow all headers
)

# Load models (Ensure correct paths for your models)
with open('C:\\Customer Journey Analysis\\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

kmeans_model = pickle.load(open(r'C:\Customer Journey Analysis\kmeans_model.pkl', 'rb'))
autoencoder = load_model(r'C:\Customer Journey Analysis\autoencoder_model.h5')

# Define input data schema for validation
class CustomerData(BaseModel):
    age: int
    income: float
    spending_score: float

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        # Input data as numpy array
        input_data = np.array([[customer.age, customer.income, customer.spending_score]])

        # Scale input data
        scaled_input = scaler.transform(input_data)

        # Encode data with autoencoder
        encoded_data = autoencoder.predict(scaled_input)

        # Print encoded data to debug
        print("Encoded Data:", encoded_data)

        # Predict cluster using K-Means
        cluster = kmeans_model.predict(encoded_data)

        return {
            "cluster": int(cluster[0]),
            "encoded_data": encoded_data.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
