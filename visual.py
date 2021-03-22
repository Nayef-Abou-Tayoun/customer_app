#from pycaret.regression import load_model, predict_model
import pickle
from xgboost import XGBClassifier
import streamlit as st
import pandas as pd
import numpy as np

load_model = open('model.pkl', 'rb')      
model = pickle.load(load_model) 
load_prepro = open('prepro.pkl', 'rb')      
prepro= pickle.load(load_prepro) 

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    #from PIL import Image
    #image = Image.open('logo.png')
    #image_hospital = Image.open('hospital.jpg')

    #st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict patient hospital charges')
    st.sidebar.success('https://www.pycaret.org')
    
    #st.sidebar.image(image_hospital)

    st.title("Customer Loyalty App")

    if add_selectbox == 'Online':

        CP = st.number_input('Customer Pay', min_value=0, max_value=10000)
        GW = st.number_input('Good Will', min_value=0, max_value=10000)
        IR = st.number_input('Interest Rate', min_value=1, max_value=10)
        NC = st.selectbox('Current No of Cars', [0,1,2,3,4])
        NP = st.selectbox('Previous No of Cars', [0,1,2,3,4])
        OM = st.number_input('Odometer_reading', min_value=0, max_value=100000000)
        MY = st.selectbox('Model_year', [2015,2016,2017,2018,2019])
        IC = st.number_input('Income', min_value=0, max_value=100000000)
        LS = st.number_input('Days since last service', min_value=0, max_value=1000)
        ag = st.number_input('Age', min_value=16, max_value=100)
        WP = st.number_input('Warranty Pay', min_value=0, max_value=100000)
        TP = st.number_input('Total monthly payment', min_value=0, max_value=100000)
        T = st.number_input('Terms', min_value=0, max_value=100)


        output=""

        input_dict = {'customer_pay': CP ,'good_will': GW, 'Rate': IR,'number_active':NC, 'number_historic_bmw': NP, 
        'odometer_reading':OM,'last_service_date':LS,'term':T,'total_monthly_payment':TP,'warranty_pay':WP,'age_Group':ag,
        'days_since_last_service':LS,'income ':IC,'model_year':MY}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            input_df_1=prepro.transform(input_df)
            input_df_2 = pd.DataFrame(input_df_1, columns = input_df.columns)
            output = model.predict(input_df_2)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = model.predict_model(data)
            st.write(predictions)



if __name__ == '__main__':
    run()