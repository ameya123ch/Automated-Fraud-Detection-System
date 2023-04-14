import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import requests
from streamlit_lottie import st_lottie

st.title("Automated Fraud Detection System Web app")
st.write("""

This app will helps us to track what type of transactions lead to fraud. I collected a dataset from [Kaggle repositry](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection)
,which contains historical information about fraudulent transactions which can be used to detect fraud in online payments.
""")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url = "https://assets8.lottiefiles.com/packages/lf20_yhTqG2.json"

lottie_hello = load_lottieurl(lottie_url)

with st.sidebar:
    st_lottie(lottie_hello,quality='high')

st.sidebar.title('Users Features Explanation')
st.sidebar.markdown("**step**: represents a unit of time where 1 step equals 1 hour")
st.sidebar.markdown("**type**: type of online transaction")
st.sidebar.markdown('**amount**: the amount of the transaction')
st.sidebar.markdown('**oldbalanceOrg**: balance before the transaction')
st.sidebar.markdown('**newbalanceOrig**: balance after the transaction')
st.sidebar.markdown('**oldbalanceDest**: initial balance of recipient before the transaction')
st.sidebar.markdown('**newbalanceDest**: the new balance of recipient after the transaction')



st.header('User Input Features')

def user_input_features():
    step = st.number_input('Step', 0, 3)
    type = st.selectbox('Online Transaction Type', ("CASH IN", "CASH OUT", "DEBIT", "PAYMENT", "TRANSFER"))
    amount = st.number_input("Amount of the transaction")
    oldbalanceOrg = st.number_input("Old balance Origin")
    newbalanceOrig = st.number_input("New balance Origin")
    oldbalanceDest = st.number_input("Old Balance Destination")
    newbalanceDest = st.number_input("New Balance Destination")
    data = {'step': step,
            'type': type,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with sample dataset
# This will be useful for the encoding phase
fraud_raw = pd.read_csv('samp_online.csv')
fraud = fraud_raw.drop(columns=['isFraud','nameOrig','nameDest','isFlaggedFraud'])
df = pd.concat([input_df,fraud],axis=0)

# Encoding of ordinal features


encode = ['type']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Reads in saved classification model
if st.button("Predict"):
    load_clf = tf.keras.models.load_model('fraud.h5', compile=False)
    load_clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




    # Apply model to make predictions

    y_probs = load_clf.predict(df)
    pred = tf.round(y_probs)
    pred = tf.cast(pred, tf.int32)

    st.markdown(
        """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 25px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    if pred == 0:

        col1, col2 = st.columns(2)
        col1.metric("Prediction", value="Transaction is not fraudulent ")
        col2.metric("Confidence Level", value=f"{np.round(np.max(y_probs) * 100)}%")
    else:
        col1, col2 = st.columns(2)
        col1.metric("prediction", value="Transaction is fraudulent")
        col2.metric("Confidence Level", value=f"{np.round(np.max(y_probs) * 100)}%")








