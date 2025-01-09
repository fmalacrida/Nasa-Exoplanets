import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# loading the trained model and images
model = joblib.load('gradient_boosting_model.pkl')
planet_images = {
    'Gas Giant': '/exoplanets_img/gasgiant-7.jpg',
    'Super Earth': '/exoplanets_img/neptunelike-8.jpg',
    'Neptune-like': '/exoplanets_img/superearth-7.jpg',
    'Terrestrial': '/exoplanets_img/terrestrial-4.jpg'
}

# feature names used in the model
features = ['mass_multiplier', 'radius_multiplier', 'stellar_magnitude', 
            'orbital_radius', 'eccentricity', 'detection_method_numeric']

# app title
st.title("Exoplanet Explorer")

# input features for prediction
st.header("Predict Planet Type")
st.write("Enter the features of the planet:")

# inputs for the features
mass = st.slider("Mass Multiplier", min_value=0.1, max_value=10.0, step=0.1)
radius = st.slider("Radius Multiplier", min_value=0.1, max_value=5.0, step=0.1)
magnitude = st.slider("Stellar Magnitude", min_value=-5.0, max_value=20.0, step=0.1)
radius_orbit = st.slider("Orbital Radius (AU)", min_value=0.1, max_value=50.0, step=0.1)
eccentricity = st.slider("Orbital Eccentricity", min_value=0.0, max_value=1.0, step=0.01)
detection_method = st.selectbox("Detection Method", options=['Radial Velocity', 'Direct Imaging', 'Transit', 'Timing', 'Eclipse Timing Variations', 'Astrometry'])

# map detection method
detection_method_mapping = {
    'Radial Velocity': 0,
    'Direct Imaging': 1,
    'Transit': 2,
    'Timing': 3,
    'Eclipse Timing Variations': 4,
    'Astrometry': 5
}
detection_method_numeric = detection_method_mapping[detection_method]

# input data
input_data = pd.DataFrame([{
    'mass_multiplier': mass,
    'radius_multiplier': radius,
    'stellar_magnitude': magnitude,
    'orbital_radius': radius_orbit,
    'eccentricity': eccentricity,
    'detection_method_numeric': detection_method_numeric
}])

# prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        # prediction to planet type
        planet_types = {0: 'Gas Giant', 1: 'Super Earth', 2: 'Neptune-like', 3: 'Terrestrial'}
        predicted_class = planet_types[prediction]
        
        # show the result
        st.success(f"The predicted planet type is: **{predicted_class}**")
        st.image(Image.open(planet_images[predicted_class]), caption=predicted_class, use_column_width=True)
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
