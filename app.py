import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
from sklearn.metrics import r2_score
import yfinance as yf

# DB Management
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS usertable(username TEXT, password TEXT)')

def add_userdata(username, password):
    c.execute('INSERT INTO usertable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM usertable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data 

def view_all_users():
    c.execute('SELECT * FROM usertable')
    data = c.fetchall()
    return data 

st.set_page_config(page_title='Stock Prediction',page_icon='📊')

def main():
    start = "2010-01-01"
    end = "2025-04-15"


    st.title('STOCK TREND PREDICTOR')
    menu = ['Home','Login','SignUp']
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.subheader('Home')
        st.subheader('Sample Data')
        user_input = st.text_input('Enter Stock Ticker:','AAPL')
        df = yf.download(user_input, start=start, end=end)
        
        # Describing Data
        st.subheader('Data from 2010 - 2023')
        st.write(df.describe())

    elif choice == "Login":
        st.subheader('Login Section')
        username = st.sidebar.text_input('User Name')
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            result = login_user(username, password)
            if result:
                st.success("Logged in as {}".format(username))

                task = st.selectbox("Task",["Add Ticker","Visualizations","Prediction"])
                if task == "Add Ticker":
                    st.subheader("Enter your Ticker:")
                    user_input = st.text_input('Enter Stock Ticker:','AAPL')
                    df = yf.download(user_input, start=start, end=end)
                    # Describing Data
                    st.subheader('Data from 2010 - 2021')
                    st.write(df.describe())
                elif task == "Visualizations":
                    st.subheader("Data Visualizations")
                    # Data Visualizations
                    user_input = st.text_input('Enter Stock Ticker:','AAPL')
                    df = yf.download(user_input, start=start, end=end)
                    st.subheader('Closing Price vs. Time chart')
                    fig = plt.figure(figsize=(12,6))
                    plt.plot(df.Close)
                    st.pyplot(fig)

                    st.subheader('Closing Price vs. Time chart with 100MA')
                    ma100 = df.Close.rolling(100).mean()
                    fig = plt.figure(figsize=(12,6))
                    plt.plot(ma100,'r')
                    plt.plot(df.Close)
                    st.pyplot(fig)

                    st.subheader('Closing Price vs. Time chart with 100MA and 200MA')
                    ma100 = df.Close.rolling(100).mean()
                    ma200 = df.Close.rolling(200).mean()
                    fig = plt.figure(figsize=(12,6))
                    plt.plot(ma100,'r')
                    plt.plot(ma200,'g')
                    plt.plot(df.Close)
                    st.pyplot(fig)
                elif task == "Prediction":
                    st.subheader("Model Prediction")
                    user_input = st.text_input('Enter Stock Ticker:','AAPL')
                    df = yf.download(user_input, start=start, end=end)
                    #Splitting the Dataframe into Training and Testing datasets

                    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
                    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))]) 

                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler(feature_range=(0,1))

                    data_training_array = scaler.fit_transform(data_training)

                    # Load my model
                    model = load_model('keras_model1.h5')

                    # Testing Part
                    past_100_days = data_training.tail(100)
                    if isinstance(past_100_days, pd.DataFrame):
                        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
                    else:
                        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
                    
                    input_data = scaler.fit_transform(final_df)

                    x_test = []
                    y_test = []

                    for i in range(100, input_data.shape[0]):
                        x_test.append(input_data[i-100:i])
                        y_test.append(input_data[i, 0])
    
                    x_test, y_test = np.array(x_test), np.array(y_test)
                    y_predicted = model.predict(x_test)
                    scaler = scaler.scale_

                    scale_factor = 1/scaler[0]
                    y_predicted = y_predicted * scale_factor
                    y_test = y_test * scale_factor

                    # Final Graph 
                    st.subheader('Original Price vs. Predicted Price(Graph)')
                    fig2 = plt.figure(figsize=(12,6))
                    plt.plot(y_test, 'b', label='Original Price')
                    plt.plot(y_predicted, 'r', label='Predicted Price')
                    plt.xlabel('Time')
                    plt.ylabel('Price')
                    plt.legend()
                    st.pyplot(fig2)
                    acc = r2_score(y_test, y_predicted)
                    st.write("The R-Squared Value is:", acc)
                    if acc > 0.8:
                        st.subheader("This is a best fit. You can Safely Invest in this stock and earn profits.😊")
                    else:
                        st.subheader("This is a worst fit. Be safe Before Investing in this stock.☹")
            else:
                st.warning("Invalid Username/Password.")
    else:
        st.subheader('Create New Account')    
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')
        if st.button("Register Here"):
            create_usertable()
            add_userdata(new_user, new_password)
            st.success("You have successfully created a New Account.")
            st.info("Go to the Login Menu in the Sidebar to Login...")

if __name__=='__main__':
    main()
