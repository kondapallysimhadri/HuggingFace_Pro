import streamlit as st
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def train_model():
    # Load California Housing dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Train the model (cached so it doesn’t retrain every button click)
model = train_model()

# Streamlit UI
st.title("🏡 House Price Prediction (California Housing)")
st.write("Enter house details and get the predicted price (in $100,000s).")

MedInc = st.number_input("Median Income (10k USD)", min_value=0.0, max_value=20.0, step=0.1)
HouseAge = st.number_input("House Age (years)", min_value=1, max_value=100, step=1)
AveRooms = st.number_input("Average Rooms", min_value=1.0, max_value=20.0, step=0.1)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.5, max_value=10.0, step=0.1)
Population = st.number_input("Population", min_value=1, max_value=10000, step=1)
AveOccup = st.number_input("Average Occupants", min_value=1.0, max_value=10.0, step=0.1)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, step=0.1)
Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, step=0.1)

if st.button("Predict Price"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(input_data)[0]
    st.success(f"🏠 Estimated House Price: ${prediction*100000:,.2f}")
