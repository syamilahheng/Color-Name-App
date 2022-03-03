import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.header("Colour Name Determination App") 

st.title("About")

st.write("""
Hi, it's Syamilah! In this Color Name Determination app, I am going to build an application through which you can automatically get the name of the color by choosing the RGB color range.

Colors are made up of 3 primary colors; red, green, and blue. In computers, we define each color value within a range of 0 to 255. So in how many ways we can define a color? The answer is 256256256 = 16,581,375. There are approximately 16.5 million different ways to represent a color.
""")

st.sidebar.header('User Input RGB Color Range')

def user_input_features():
    red = st.sidebar.slider('Red', 0, 255, 100)
    green = st.sidebar.slider('Green', 0, 255, 100)
    blue = st.sidebar.slider('Blue', 0, 255, 100)
    data = {'red': red,
            'green': green,
            'blue': blue}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input RGB Color Range')
st.write(df)

colors = pd.read_csv("https://raw.githubusercontent.com/syamilahheng/Color-Name-App/main/colors.csv")
X = data.drop(['yyy'], axis = 1)
Y = colors.yyy

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Color Determination')
st.write(colors.yyy[prediction])
# st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)