import pandas as pd
import streamlit as st
import pickle
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
    data = pd.read_csv("data.csv")
    data = data.drop(["Unnamed: 32", "id", 
                      "radius_mean", "perimeter_mean", "compactness_mean", "concave points_mean", 
                       "radius_se", "perimeter_se", "compactness_se", "concave points_se",
                       "radius_worst","perimeter_worst", "compactness_worst", "concave points_worst"] ,axis = 1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def add_sidebar():
    st.sidebar.header("Cell Nuclei Values")

    data = get_clean_data()

    slider_labels = [
        ("Texture (mean)", "texture_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        
        ("Texture (se)", "texture_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        
        ("Texture (worst)", "texture_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),       
        ("Concavity (worst)", "concavity_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst")
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value = float(0),
            max_value= float(data[key].max()),
            value = float(data[key].mean())
        )

    return input_dict


def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(["diagnosis"], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] =  scaled_value
    
    return scaled_dict


def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data) 
    categories = ["Texture", "Area", "Smoothness", "Concavity", "Symmetry", "Fractal Dimension"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data["texture_mean"], input_data["area_mean"], input_data["smoothness_mean"], 
            input_data["concavity_mean"], input_data["symmetry_mean"], input_data["fractal_dimension_mean"]
        ],
        theta=categories,
        fill="toself",
        name="Mean Value"
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data["texture_se"], input_data["area_se"], input_data["smoothness_se"],
            input_data["concavity_se"], input_data["symmetry_se"], input_data["fractal_dimension_se"]
        ],
        theta=categories,
        fill="toself",
        name="Standard Error"
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data["texture_worst"], input_data["area_worst"], input_data["smoothness_worst"], 
            input_data["concavity_worst"], input_data["symmetry_worst"], input_data["fractal_dimension_worst"]
        ],
        theta=categories,
        fill="toself",
        name="Worst Value"
    ))
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=True,
       width = 650,  # Adjust width as needed
       height = 650  # Adjust height as needed
    )
    return fig


def add_predictions(input_data):
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)
    prediction_proba = model.predict_proba(input_array_scaled)

    st.subheader("Prediction:")
    st. write("")
    
    if prediction[0] == 0:
        st.write("**The cell tumor is :green[Benign]**")
    else:
        st.write("**The cell tumor is :red[Malicious]**")

    st. write("")

    # Convert probabilities to percentatges:
    benign_proba_percentage = prediction_proba[0][0] * 100
    malicious_proba_percentage = prediction_proba[0][1] * 100
    
    st.write(f"Probability of being benign:\n **:green[{benign_proba_percentage:.2f}%]**")
    st.write(f"Probability of being malicious:\n **:red[{malicious_proba_percentage:.2f}%]**")

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")



def main():
    st.set_page_config(
        page_title= "Breast Cancer Predictor",
        page_icon = ":female-doctor:",
        layout= "wide",
        initial_sidebar_state="expanded"
    )

    input_data = add_sidebar()

    # Add an image:
    col1, col2, col3 = st.columns([2, 5, 2])
    with col2:
       st.image("foto.jpeg", width = 750)

    # Add title and description:
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This app predicts using a machine learning model (logistic regression) whether a breast mass is benign or malignant, based on the measurements values of the nuclei cell introduced in the slidebar.")

    # Add radar chart and the predictions from our model:
    col4, col5, col6 = st.columns([1, 4, 2])

    with col5:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        
    with col6:
        add_predictions(input_data)

# To check that we read from the correct "main" file before run the function
if __name__ == '__main__':
    main()