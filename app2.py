import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt
#import shap
import pickle
from xgboost import XGBClassifier

def app():
    st.title("Customer Loyalty Insight Dashboard")
    df = pd.read_csv('data_2_ana.csv')
    df['count']=1

    #add_selectbox = st.selectbox(
    #"Which data would you like to use ?",
    #("Stored data", "Upload New Data"))


    #file_upload_1 = pd.read_csv('data_2_ana.csv')
          
    #df= pd.read_csv(file_upload_1)
    df['count']=1
    


    #load_model = open('model.pkl', 'rb')      
    #model = pickle.load(load_model)    





    #predictions = model.predict_model(data)
    #st.write(predictions)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    # get the dataset and read dataset into

    #df = pd.read_csv('data')


    fig = px.pie(df, values='count', names='Loyal', title='Total Customers')
    st.plotly_chart(fig,use_container_width=True)

    fig_1 = px.pie(df, values='Total_monthly_payment', names='Loyal', title='Total Monthly Payments ($) ')
    st.plotly_chart(fig_1,use_container_width=True)

    
    fig_2 = px.pie(df, values='Customer_cost', names='Loyal', title='Customer Cost per Year ($)')

    st.plotly_chart(fig_2,use_container_width=True)


    d=df.groupby(by=["Loyal"]).mean().round(0)
    

    fig6 = go.Figure(go.Indicator(mode = "gauge+number",
    value = d['Warranty_pay']['Loyal'],
    title = {'text': "Loyal Customer's average warranty pay per year"},
    domain = {'x': [0, 1], 'y': [0, 1]}))
    

    st.plotly_chart(fig6)

    fig77 = go.Figure()
    fig77.add_trace(go.Indicator(
    mode = "number+delta",
    value = d['Warranty_pay']['Disloyal'],
    title = {"text": "Disloyal Customer's average warranty pay per year<br><span style='font-size:0.8em;color:gray'>Warranty pay difference compared to loyal customers (%)</span><br>"},
    delta = {'reference': d['Warranty_pay']['Loyal'], 'relative': True},
    domain = {'x': [0, 1], 'y': [0, 1]}))

    st.plotly_chart(fig77)


    service_loyal = d['Average_service_pay']['Loyal']
    service_disloyal = d['Average_service_pay']['Disloyal']
  
    fig6 = go.Figure(go.Indicator(mode = "gauge+number",
    value = service_loyal,
    title = {'text': "Loyal Customer's average service pay per visit"},
    domain = {'x': [0, 1], 'y': [0, 1]}))
    

    st.plotly_chart(fig6)

    fig77 = go.Figure()
    fig77.add_trace(go.Indicator(
    mode = "number+delta",
    value = service_disloyal,
    title = {"text": "Disloyal Customer's average service pay per visit<br><span style='font-size:0.8em;color:gray'>Service pay difference compared to loyal customers (%)</span><br>"},
    delta = {'reference': service_loyal, 'relative': True},
    domain = {'x': [0, 1], 'y': [0, 1]}))

    st.plotly_chart(fig77) 


    #sns.heatmap(df_X.corr(method='pearson'), cmap='terrain', linewidths=0.1)
    #st.pyplot()
    
    fig_3=sns.boxplot(x=df['Total_monthly_payment'], y=df['Loyal'], palette="Set3",showmeans=True ).set(
    xlabel='Monthly Payment ($)', 
    ylabel='',
    xticks=range(0,4800,380))
    st.pyplot()


    fig_4=sns.boxplot(x=df['Term'], y=df['Loyal'], palette="Set3",showmeans=True).set(
    xlabel='Payments Terms (Months)', 
    ylabel='',
    xticks=range(0,100,5))
    st.pyplot()
    
    
    #sns.pairplot(df, hue='loyal_or_not')
    #st.pyplot()

    #d=df.groupby(by=["loyal_or_not"]).mean().round(0)
    #st.write(df)




if __name__ == '__main__':
    app()
