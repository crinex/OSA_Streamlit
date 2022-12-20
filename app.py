import streamlit as st
import pandas as pd
import numpy as np
# from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

st.set_page_config(
    layout="centered", page_icon='🛌', page_title="Predict OSA App"
)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("""
# 🛌 Sleep Apnea Score Prediction
This app predicts the **Sleep Apnea's RDI Score**!
""")
X = pd.read_csv('data.csv')
X = X.drop(['날짜', 'PSG 번호', 'PSG종류', '병록번호', '이름', '진단명', 'original_path', 'index', 'index_path', 'RDI_label', '주진단분류'], axis=1)
X['RDI'] = X['RDI'].fillna(X['RDI'].mean())
X['PLMI'] = X['PLMI'].fillna(X['PLMI'].mean())
X['ESS'] = X['ESS'].fillna(X['ESS'].mean())
Y = X['RDI']
X.drop('RDI', axis=1, inplace=True)
X.drop('Unnamed: 0', axis=1, inplace=True)

st.sidebar.header('Specify Input Parameters')
reg = st.sidebar.selectbox('Choose one Regression Model', ('MLP', 'RF'))
def user_input_features():
    HT = st.sidebar.slider('Height', 1.20, 2.00, 1.50)
    WT = st.sidebar.slider('Weight', 30., 180., 70.)
    SEX = st.sidebar.selectbox('Choose your Sex (Male / Female)',('M', 'F'))
    AGE = st.sidebar.slider('Age', 8, 90)
    PLMI = st.sidebar.slider('PLMI', 0., 120., 54.)
    ESS = st.sidebar.slider('ESS', 0., 25., 8.43)
    BMI = WT / (HT*HT)
    data = {
            'Sex': SEX,
            'Age': AGE,
            'Ht': HT,
            'Wt': WT,
            'BMI': BMI,
            'PLMI': PLMI,
            'ESS': ESS,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.header('Specified Input parameters')
st.write(df)
st.write('---')
df['Sex'] = df['Sex'].replace({'F':0, 'M':1})
X['Sex'] = X['Sex'].replace({'F':0, 'M':1})

if reg == 'MLP':
    model = MLPRegressor()
    model.fit(X,Y)
    explainer = shap.KernelExplainer(model.predict, df)
    shap_values = explainer.shap_values(X)
elif reg == 'RF':
    model = RandomForestRegressor()
    model.fit(X,Y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)


prediction = model.predict(df)

grade = ''
cr = ''
if prediction < 5:
    grade = 'Normal'
    cr = 'white'
elif prediction >= 5 and prediction < 15:
    grade = 'Mild'
    cr = 'green'
elif prediction >= 15 and prediction < 30:
    grade = 'Moderate'
    cr = 'blue' 
elif prediction >= 30:
    grade = 'Severe'
    cr = 'red'

st.header('Prediction of RDI Score')
st.write(prediction)
st.markdown(f"### Your OSA Grade is ***:{cr}[{grade}]***")
st.write('---')

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
