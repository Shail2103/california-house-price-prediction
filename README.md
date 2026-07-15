# 🏡 California House Price Predictor

An end-to-end Machine Learning project that predicts the median house value in California districts using **Scikit-Learn** and provides an interactive web dashboard built with **Streamlit**.

##  Features
* **Machine Learning Pipeline:** Uses a `RandomForestRegressor` pre-processed with `StandardScaler` to handle feature scaling and robust predictions.
* **Interactive UI:** A clean, web-based dashboard allowing users to input district features using sliders and text inputs.
* **Dynamic Mapping:** Features a built-in map visualizer that displays the target district's geographical location based on the inputted latitude and longitude.

##  Dataset
The project uses the classic **California Housing Dataset** natively sourced from Scikit-Learn (`sklearn.datasets.fetch_california_housing`). Features include:
* Median Income
* House Age
* Average Rooms / Bedrooms
* Population
* Average Occupancy
* Latitude & Longitude
