import streamlit as st

import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

loaded_model = pickle.load(open('Wine_model.sav','rb'))

def check(input_data):

    array_input = np.array(input_data)

    reshaped_input = array_input.reshape(1,-1)

    prediction = loaded_model.predict(reshaped_input)

    return "{:.2f}".format(prediction[0]*100)

def main():
    st.title("Red Wine Prediction")

    fixed_acidity	 = st.number_input("fixed_acidity")

    volatile_acidity	 = st.number_input("volatile_acidity")

    citric_acid	 = st.number_input("citric_acid")

    residual_sugar	 = st.number_input("residual_sugar")

    chlorides = st.number_input("chlorides")

    free_sulfur_dioxide	 = st.number_input("free_sulfur_dioxide")

    total_sulfur_dioxide	 = st.number_input("total_sulfur_dioxide")

    density	 = st.number_input("density")

    pH	 = st.number_input("pH")

    sulphates	 = st.number_input("sulphates")

    alcohol	 = st.number_input("alcohol")

    pred = ""
    if st.button("Click Here for Red Wine Prediction"):
        pred = check([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])

    st.success(f"Wine Quality Test is {pred} %")

if __name__=='__main__':
    main()
    