import streamlit as st
import joblib
import pandas as pd

model = joblib.load("tesla_forecasting_model.pkl")

st.title("Tesla Deliveries Forecasting")

year = st.number_input("Year", 2015, 2030, 2025)
month = st.number_input("Month", 1, 12, 1)

production_units = st.number_input("Production Units", 1000, 50000, 10000)

avg_price = st.number_input("Average Price USD", 10000.0, 200000.0, 50000.0)

battery = st.number_input("Battery Capacity (kWh)", 50, 200, 75)

range_km = st.number_input("Range (km)", 200, 1000, 500)

co2_saved = st.number_input("CO2 Saved Tons", 1000, 500000, 50000)

charging = st.number_input("Charging Stations", 1, 10000, 500)

if st.button("Predict"):

    data = pd.DataFrame({
        'Year':[year],
        'Month':[month],
        'Region':[0],
        'Model':[0],
        'Production_Units':[production_units],
        'Avg_Price_USD':[avg_price],
        'Battery_Capacity_kWh':[battery],
        'Range_km':[range_km],
        'CO2_Saved_tons':[co2_saved],
        'Source_Type':[0],
        'Charging_Stations':[charging],
        'Production_Efficiency':[1],
        'Cost_Per_KM':[avg_price/range_km]
    })

    prediction = model.predict(data)

    st.success(
        f"Predicted Deliveries: {int(prediction[0])}"
    )