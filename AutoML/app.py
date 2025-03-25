#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 14:48:44 2020

@author: harbhajan
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.graph_objs import Pie, Layout,Figure
import base64
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


################################################################################
image = Image.open("datadoobies_logo.PNG")
st.sidebar.image(image, use_column_width=True)
st.title('DataDoobies AutoML Tool')
st.subheader('Here, Upload your file to get information:')
st.warning('File should be less than 20 MB for better performance.')

##############
file_buffer = st.file_uploader("Please Upload a Dataset before further proceedings:",type=['csv'])#,encoding='ISO-8859-1'
#text_io = io.TextIOWrapper(file_buffer)
st.set_option('deprecation.showfileUploaderEncoding', False)
def load_data(file_buffer):
  return (pd.read_csv(file_buffer,encoding='utf-8'))
if file_buffer:
  temp = load_data(file_buffer)
# else:
#   return "Please upload a Dataset!"
#df = pd.read_csv("/content/Iris.csv")
# if st.checkbox('Show dataframe'):
  st.subheader('Your Uploaded Dataset!')
  st.write(temp)
  st.success('Successfully, You have uploaded file!')
  st.sidebar.success('Now select the below checkbox for more details:')    
########################################################################################################  
    #else:
    # st.subheader('Data File not found! Please Upload a Data File.')
  if st.sidebar.checkbox('Show Summary'):
    st.write("Here is statistical summary of dataset:")
    st.write(temp.describe())
  ################
    if st.sidebar.checkbox('Checking Null Values'):
      missing_values_count = temp.isnull().sum()
      st.subheader('Percentage missing of Data in each column:')
      missing_values_percent = round((100*(temp.isnull().sum()/len(temp))),2)
      st.write(pd.DataFrame({'% Missing Values of each Column ': missing_values_percent}))
      # st.write(missing_values_percent)
    ################
      #how many total missing values
      total_cells = np.product(temp.shape)
      total_missing = missing_values_count.sum()

      # # percent of data that is missing
      # st.text('Total Missing percentage of data:')
      # st.write((total_missing/total_cells) * 100)
      st.write("Number of Rows : ",len(temp))
      st.write("Number of Columns : ",len(temp.columns))
      st.write("Number of total Cells : ",total_cells)
      st.write("Number of missing cells :  ",total_missing)
      st.write("Total missing percentage of data : ",round(((total_missing/total_cells) * 100),2),"%")
    ################
      if st.sidebar.checkbox("Show datatypes of the columns"):
        st.subheader('Here are the datatypes of each column:')
        st.write(temp.dtypes)
        if st.sidebar.success('Treating with Null Values'):
          st.subheader('Here are some methods to remove null values:')
      # impute = SimpleImputer(missing_values=nan, strategy='mean')
      # transformed_data = impute.fit_transform(temp)
      #trans_data = pd.DataFrame(MICE().fit_transform(temp))
      #transformed_data = trans_data.isnull().sum()
      # Select a column to treat missing values
          #######################################################################################################
          col_option = st.sidebar.multiselect("Select Column to treat missing values by Mean", temp.columns)
          if st.success("Replace with Mean"):
            replaced_value = temp[col_option].mean()
            st.write("Mean value of column is :", pd.DataFrame({'Mean Value of Each Column': replaced_value}))
            temp[col_option] = temp[col_option].fillna(temp[col_option].mean())
            if st.checkbox("Apply to Replace with Mean"):  
              st.write(temp)
          col_option = st.sidebar.multiselect("Select Column to treat missing values by Median", temp.columns)
          if st.success("Replace with Median"):
            replaced_value = temp[col_option].median()
            st.write("Median value of column is :", pd.DataFrame({'Median Value of Each Column': replaced_value}))
            temp[col_option] = temp[col_option].fillna(temp[col_option].median())
            if st.checkbox("Apply to Replace with Median"):  
              st.write(temp)
          col_option = st.sidebar.multiselect("Select Column to treat missing values by Mode", temp.columns)
          if st.success("Replace with Mode"):
            replaced_value = temp[col_option].mode()
            st.write("Mode value of column is :", replaced_value)
            temp[col_option] = temp[col_option].fillna(temp[col_option].mode().iloc[0], inplace=False)
            if st.checkbox("Apply to Replace with Mode"):  
              st.write(temp)
          #######################################################################################################
          # col_option = st.sidebar.multiselect("Select Column to treat missing values", temp.columns) 

          # # Specify options to treat missing values
          # missing_values_clear = st.sidebar.selectbox("Select Missing values treatment method", ("Replace with Mean", "Replace with Median", "Replace with Mode"))
        
          # if missing_values_clear == "Replace with Mean":
          #   replaced_value = temp[col_option].mean()
          #   st.write("Mean value of column is :", replaced_value)
          #   temp[col_option] = temp[col_option].fillna(temp[col_option].mean())
          # elif missing_values_clear == "Replace with Median":
          #   replaced_value = temp[col_option].median()
          #   st.write("Median value of column is :", replaced_value)
          #   temp[col_option] = temp[col_option].fillna(temp[col_option].median())
          # elif missing_values_clear == "Replace with Mode":
          #   replaced_value = temp[col_option].mode()
          #   st.write("Mode value of column is :", replaced_value)
          #   temp[col_option] = temp[col_option].fillna(temp[col_option].mode())
          # st.write(temp)
          # # st.write(temp)
          st.subheader("Missing values are replaced. Now, You can download the file without null values.")
          
          ##########################################################################################################  
          # if st.checkbox('Download File'):
          # df = pd.DataFrame(data, columns=["Col1", "Col2", "Col3"])
          csv = temp.to_csv(index=True)
          b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
          href = f'<a href="data:file/csv;base64,{b64}" download="myoutputfile.csv">Download file</a>'
          st.markdown(href, unsafe_allow_html=True)
    ###########################################################################################################
    # To change datatype of a column in a dataframe
    # display datatypes of all columns
      # feature = st.selectbox('Select feature:', temp.columns[:])
      # visualization
    # scatter plot
        if st.sidebar.success("Exploratory Data Analysis"):
          if st.sidebar.checkbox('Data Visualization'):
            st.subheader('Here, some basic plots for dataset:')
            basic_visual = st.selectbox("Select data visualization Plot!", ("Bar Chart", "Line Chart", "Bubble Chart", "Pie Chart", "Scatterplot", "Boxplot"))

              ############################ Histogram ########
            if basic_visual == "Bar Chart":
              col1 = st.selectbox('Select feature on x', temp.columns)
              col2 = st.selectbox('Select feature on y', temp.columns)
              if st.checkbox("Show Plot"):
                  fig = px.bar(temp, x =col1,y=col2)
                  st.plotly_chart(fig)
                  
              ########################### Line Chart ########
            elif basic_visual == "Line Chart":
              col1 = st.selectbox('Select feature on x', temp.columns)
              col2 = st.selectbox('Select feature on y', temp.columns)
              if st.checkbox("Show Plot"):
                  fig = px.line(temp, x =col1,y=col2)
                  st.plotly_chart(fig)
                  
              ########################### BUBBLE Plot ###########
            elif basic_visual == "Bubble Chart":
              col1 = st.selectbox('Select feature on x', temp.columns)
              col2 = st.selectbox('Select feature on y', temp.columns)
              if st.checkbox("Show Plot"):
                  fig = px.scatter(temp, x =col1,y=col2, log_x=True, size_max=60)
                  st.plotly_chart(fig)
                  
              ########################### Pie Chart #########
            elif basic_visual == "Pie Chart":
              col1 = st.selectbox('Select feature for PieChart', temp.columns)
                #fig = px.pie(temp, values=temp[col1], color_discrete_sequence=px.colors.sequential.RdBu)
              temp_df = pd.DataFrame(list(temp[col1].value_counts().to_dict().items()))
              if st.checkbox("Show Plot"):
                  fig = px.pie(values=temp_df[1], names=temp_df[0])                
                  st.plotly_chart(fig)
                  
              ########################### Scatter Plot ######
            elif basic_visual == "Scatterplot":
              col1 = st.selectbox('Select feature on x?', temp.columns)
              col2 = st.selectbox('Select feature on y?', temp.columns)
              if st.checkbox("Show Plot"):
                  fig = px.scatter(temp, x =col1,y=col2)
                  st.plotly_chart(fig)
                  
            ########################### BOX PLOT##########      
            elif basic_visual == "Boxplot":
              col1 = st.selectbox('Select feature for boxplot:', temp.columns)
              if st.checkbox("Show Plot"):
                  fig = px.box(temp, y=col1)
                  st.plotly_chart(fig)
          ######################################################################
          ##Distribution Plot
          if st.sidebar.checkbox("Show Distribution plots"):
            st.write("Here is Distribution plot for selected column:")
            sel_col = st.selectbox("Select column for dsitribution plot:", temp.columns)
            st.write(sns.distplot(temp[sel_col]))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

          # correlartion plots##################################################
          if st.sidebar.checkbox("Show Correlation plots"):
            #   st.write(sns.heatmap(temp.corr()))
            #   st.set_option('deprecation.showPyplotGlobalUse', False)
            #   st.pyplot()
            # if st.checkbox("Correlation"):
            st.write("Here is correlation plot:")
            Var_Corr = temp.corr()
              # plot the heatmap and annotation on it
            st.write(sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

          if st.sidebar.checkbox("Show Correlation Scatter Plot"):
            #   st.write(sns.heatmap(temp.corr()))
            #   st.set_option('deprecation.showPyplotGlobalUse', False)
            #   st.pyplot()
            # if st.checkbox("Correlation"):
            st.write("Here is correlation scatter plot:")
            # Var_Corr = temp.corr()
              # plot the heatmap and annotation on it
            st.write(sns.pairplot(temp))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
          ######################################################################
          if st.sidebar.success("Encoding Methods"):
            if st.sidebar.checkbox('Label encoding'):
              st.write("Basically, this method is used for categorical column to convert into numeric values.")
              # encod_meth = st.selectbox("Select encoding method!", ("Label Encoding", "One Hot Encoding"))
                  # creating initial dataframe
              col_option = st.multiselect("Select Column to treat Label Encoding", temp.columns)
              if st.checkbox("Apply Label Encoding"):  
                modify_dfs = pd.DataFrame(temp, columns=col_option)
                  # creating instance of labelencoder
                labelencoder = LabelEncoder()
                  # Assigning numerical values and storing in another column
                #modify_df['modify_df_Cat'] = labelencoder.fit_transform(modify_df[col_option])
                modify_dfs = modify_dfs.apply(LabelEncoder().fit_transform)
                modify_df_drop = temp.drop(col_option, axis=1)
                modify_df = modify_df_drop.join(modify_dfs)
                st.write(modify_df)

            if st.sidebar.checkbox('One Hot encoding'):
              st.write("Basically, this method is used for categorical column to convert into numeric values.")
              col_option = st.multiselect("Select Column to treat One Hot Encoding", temp.columns)
              if st.checkbox("Apply One Hot Encoding"):
                modify_df = pd.DataFrame(temp, columns=col_option)
                  #####################################################################################################
                dum_df = pd.get_dummies(modify_df, columns=col_option)
                  # merge with main df bridge_df on key values
                modify_df = temp.drop(col_option, axis=1)
                modify_df = modify_df.join(dum_df)
                st.write(modify_df)
          #########################
          ##ML MODELS######
          if st.sidebar.success("ML Model & Deployment"):
            if st.sidebar.checkbox("Machine Learning Models"):
              st.warning('This part requires some basic machine learning knowledge!')
              model = st.selectbox("Please Choose ML Model", ("Regression", "Classification"))
              x = st.slider('Choose percentage ratio between 60%-100% to split the dataset into Training & Testing Dataset', 60.0, 100.0)
              if model == "Regression":
                drop_list = st.multiselect("Choose features which you want to remove from training dataset:", modify_df.columns)
                X = modify_df.drop(drop_list, axis=1)
                targ_list = st.selectbox("Choose Target feature:", modify_df.columns)
                y = modify_df[targ_list]
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=x/100, random_state=42)
                if st.button("Apply ML Model"):
                  lin_model = LinearRegression()
                  lin_model.fit(X_train, y_train)
                  Y_pred = lin_model.predict(X_test)
                  st.write(pd.DataFrame({'Predicted Result on Test Data': Y_pred}))
                  st.write('Accuracy of regression model is', (lin_model.score(X_test, y_test)*100),'%')
                  
              elif model == "Classification":
                temp = temp.sample(frac=1).reset_index(drop=True)
                drop_list = st.multiselect("Choose features which you want to remove from training dataset:", modify_df.columns)
                X = modify_df.drop(drop_list, axis=1)
                targ_list = st.selectbox("Choose Target feature:", modify_df.columns)
                y = modify_df[targ_list]
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=x/100, random_state=42)
                if st.button("Apply ML Model"):
                  lin_model = LogisticRegression(random_state=0)
                  lin_model.fit(X_train, y_train)
                  Y_pred = lin_model.predict(X_test)
                  st.write(Y_pred)
                  st.write(confusion_matrix(y_test,Y_pred))
                  st.write("Accuracy of classification model is", accuracy_score(y_test,Y_pred), '%')
                  st.write("Classification Report:", classification_report(y_test,Y_pred))

