import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
# from prediction import predict


st.title('Predict your Sleep Apnea Score')
st.markdown('Toy model to predict your Sleep Apnea Score')

st.header("Sleep Apnea Features")
col1, col2, col3 = st.columns(3)

sc = StandardScaler()
result = 0

def predict(data):
    clf = load_model("osa_reg2.h5")
    return clf.predict(data)

with col1:
    # st.text('Height (m)')
    height = st.number_input('Your Height(m), honestly')

    sex = st.selectbox(
        'Choose your Sex (Male / Female)',
        ('Male', 'Female')
    )
    if sex is 'Male':
        sex = 0
    else:
        sex = 1

    plmi = st.number_input('PLMI Score')

with col2:
    # st.text('Weight (kg)')
    weight = st.number_input('Your Weight(kg), honestly')

    ess = st.number_input('ESS Score')
    age = st.number_input('Your Age')
    bmi = weight / (height*height)
    st.write('Your BMI is ', bmi)

with col3:
    if st.button('Predict'):
        data = np.array([age, sex, height, weight, bmi, plmi, ess])
        data = data.reshape(1, -1)
        data  = sc.fit_transform(data)
        result = predict(data)
        # result = predict(
        # np.array([[age, sex, height, weight, bmi, plmi, ess]]))
    st.text(result)
    # height = st.number_input('Your Height, honestly')

# st.text('')
# if st.button("Predict type of Iris"):
#     result = predict(
#         np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
#     st.text(result[0])
