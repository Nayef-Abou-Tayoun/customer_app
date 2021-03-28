#from pycaret.regression import load_model, predict_model
import pickle
import xgboost
import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = pickle.load(open("pima.pickle.dat", "rb"))
load_prepro = open('prepro.pkl', 'rb')      
prepro= pickle.load(load_prepro) 
def app():


    st.title("Customer Loyalty App")
    ag = st.number_input('Age', min_value=16, max_value=100)
    IC = st.number_input('Income', min_value=0, max_value=100000000)
    CM = st.checkbox('Commercial electronic messages')
    if CM : 
        CM  = 1
    else: 
        CM = 0 
    IR = st.number_input('Interst Rate', min_value=1, max_value=10)
    MP = st.number_input('Total monthly payment', min_value=0, max_value=100000)
    T = st.number_input('Term', min_value=0, max_value=100)
    WP = st.number_input('Warranty Pay', min_value=0, max_value=100000)
    AC = st.selectbox('Current No of Cars', [0,1,2,3,4])
    UN = st.checkbox('Current car is New')
    if UN : 
        UN  = 1
    else: 
        UN = 0 
    HC = st.selectbox('Previous No of Cars', [0,1,2,3,4]) 
    SP = st.number_input('Average service pay', min_value=0, max_value=10000)
    CC = st.number_input('Customer cost', min_value=0, max_value=10000) 
    LS = st.number_input('Days since last service', min_value=0, max_value=1000)
    ST = st.number_input('Service time', min_value=0, max_value=100)
    output=""

    input_dict = {'Average_service_pay':SP,'Service_time': ST, 'Commercial_electronic_messages': CM,'Customer_cost':CC, 'Rate': IR, 
    'Number_active_cars':AC,'number_historic_cars':HC,'Income':IC,'Term':T,'Total_monthly_payment':MP,'Warranty_pay':WP,'Age':ag,
    'Days_since_last_service':LS,'Used_or_New':UN}
    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        input_df_1=prepro.transform(input_df)
        input_df_2 = pd.DataFrame(input_df_1, columns = input_df.columns)
        output = model.predict(input_df_2)
        if output == 1:
            output = str('Disloyal')
        else:
            output = str('Loyal')

        output = str(output)

    st.success('The customer is {}'.format(output))
if __name__ == '__main__':
    app()