import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load or train a sample model
@st.cache_resource
def load_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, iris

model, iris = load_model()

# Streamlit App
st.title("🌸 Iris Flower Classifier 🌼")
st.write("This app uses a Random Forest model to predict the **species** of an Iris flower.")

# Sidebar input features
st.sidebar.header("Enter Flower Measurements:")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

st.subheader("Prediction:")
st.write(f"🌼 The predicted species is: **{iris.target_names[prediction][0]}**")

st.subheader("Prediction Probabilities:")
proba_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.write(proba_df)

# Visualization
st.subheader("📊 Prediction Probability Bar Chart:")
fig, ax = plt.subplots()
sns.barplot(x=proba_df.columns, y=proba_df.iloc[0], palette="pastel", ax=ax)
ax.set_ylabel("Probability")
st.pyplot(fig)
