import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

s = pd.read_csv("C:/Users/ryhan/Documents/Prog 2/social_media_usage.csv")


def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

ss = pd.DataFrame({
    "sm_li": np.where(s["web1h"] == clean_sm(s.web1h), 1, 0),
    "inc": np.where(s["income"] > 9, np.nan, s["income"]),
    "ed": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent": np.where(s["par"] == 1, 1, 0),
    "married": np.where(s["marital"] == 1, 1, 0),
    "gen_id": np.where(s["gender"] == 1, 1, 0),
    "yrs": np.where(s["age"] > 98, np.nan, s["age"])
})

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["yrs", "ed", "inc", "parent","married","gen_id"]]


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987) 


lr = LogisticRegression(class_weight = "balanced")

lr.fit(X_train, y_train)


st.title("LinkedIn User Predictor")

yrs = st.slider('How old are you?', 18, 97, 50)


ed = st.slider('Education Level between 1-7', 1, 7, 4)

inc = st.slider('What is your income level on a scale of 1-8?', 1, 8, 4)

parent = st.select_slider('Are you a parent? ',
                          
                        options = ["Yes", "No"])

if parent == "Yes":
    parent = 1
else:
    parent = 0

married = st.select_slider('Are you married?', 
                                 options = ["Yes", "No"])
if married == "Yes":
    married = 1
else:
    married = 0

gen_id = st.select_slider('Are you male or female?', 
                          options= ["Yes", "No"])

if gen_id == "Yes":
    gen_id = 1
else:
    gen_id = 0

data = (yrs,ed,inc,parent,married,gen_id)

predicted_class_1 = lr.predict([data])
if predicted_class_1 == 1:
    predicted_class_1 = "a predicted LinkeIn user"
else:
    predicted_class_1 = "not a predicted LinkedIn user"

probs_1 = lr.predict_proba([data])

probs_1[0][0]= round(probs_1[0][0],2)

probs_1[0][1]= round(probs_1[0][1],2)

st.write ("You are ", predicted_class_1)

st.write ("There is  a", probs_1[0][0], "probability  that you are NOT a LinkedIn user")

st.write ("There is  a", probs_1[0][1], "probability  that you are a LinkedIn user")