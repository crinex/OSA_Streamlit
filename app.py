import streamlit as st
import pandas as pd
import numpy as np
# from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("""
# Sleep Apnea Score Prediction App
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
# st.write(X.head())
# st.title('Predict your Sleep Apnea Score')
# st.markdown('Toy model to predict your Sleep Apnea Score')
st.sidebar.header('Specify Input Parameters')
# X.Ht.min(), X.Ht.max(), X.Ht.mean()
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
# st.write('Your BMI is ', df['BMI'])
st.write('---')
df['Sex'] = df['Sex'].replace({'F':0, 'M':1})

X['Sex'] = X['Sex'].replace({'F':0, 'M':1})
model = RandomForestRegressor()
model.fit(X,Y)

prediction = model.predict(df)

grade = ''
if prediction < 5:
    grade = 'Normal'
elif prediction >= 5 and prediction < 15:
    grade = 'Mild'
elif prediction >= 15 and prediction < 30:
    grade = 'Moderate' 
elif prediction >= 30:
    grade = 'Severe'

st.header('Prediction of RDI Score')
st.write(prediction)
st.write(f"### Your OSA Grade is ***{grade}***")
st.write('---')

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
# df['SEX'] = 
# st.header("Sleep Apnea Features")
# col1, col2, col3 = st.columns(3)

# sc = StandardScaler()
# result = 0

# def predict(data):
#     clf = load_model("osa_reg2.h5")
#     return clf.predict(data)

# with col1:
#     # st.text('Height (m)')
#     height = st.number_input('Your Height(m), honestly')

#     sex = st.selectbox(
#         'Choose your Sex (Male / Female)',
#         ('Male', 'Female')
#     )
#     if sex is 'Male':
#         sex = 0
#     else:
#         sex = 1

#     plmi = st.number_input('PLMI Score')

# with col2:
#     # st.text('Weight (kg)')
#     weight = st.number_input('Your Weight(kg), honestly')

#     ess = st.number_input('ESS Score')
#     age = st.number_input('Your Age')
#     bmi = weight / (height*height)
#     st.write('Your BMI is ', bmi)

# with col3:
#     if st.button('Predict'):
#         data = np.array([age, sex, height, weight, bmi, plmi, ess])
#         data = data.reshape(1, -1)
#         data  = sc.fit_transform(data)
#         result = predict(data)
        # result = predict(
        # np.array([[age, sex, height, weight, bmi, plmi, ess]]))
    # st.text(result)
    # height = st.number_input('Your Height, honestly')

# st.text('')
# if st.button("Predict type of Iris"):
#     result = predict(
#         np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
#     st.text(result[0])
