
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import metrics
from keras.models import load_model
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from blinker import Signal
import streamlit as st
import click
from datetime import datetime, timedelta
from streamlit import caching
caching.clear_cache()

def main():
    
    st.sidebar.title("Stock Price Predictor")
    
    # To plot the current stock price graph
    def run_process(x_value):
        df = pull_data(x_value)
        # Visualize the closing price history
        plt.figure(figsize=(16,8))
        plt.title('Close Price History')
        plt.plot(df['Close'])
        plt.xlabel('Date', fontsize =18)
        plt.ylabel('Close Price USD', fontsize = 18)
        st.pyplot()

    # To get the stock price
    def pull_data(x_value):
        mylist = {"Apple" : "AAPL", "Amazon" : "AMZN", "Tesla":"TSLA"}
        ticker = yf.Ticker(mylist[x_value])
        df = ticker.history(period="max", interval='1d')
        return df
    
    # To find the nex predicted date is a weekday - Mon to Fri as the stock market is open during weekdays
    def date_weekend():
        if datetime.today().isoweekday() in [1,2,3,4,5]:
            date_tomm = datetime.today() + timedelta(days=1)
            if date_tomm.isoweekday() in [1,2,3,4,5]:
                #print("+1 WK day",date_tomm)
                return date_tomm
            else:
                date_tomm = datetime.today() + timedelta(days=3)
                #print("+3 day",date_tomm)
                return date_tomm
        elif (datetime.today() + timedelta(days=1)).isoweekday() == 7:
            date_tomm = datetime.today() + timedelta(days=2)
            #print("+1 day",date_tomm)
            return date_tomm
        else:
            date_tomm = datetime.today() + timedelta(days=2)
            #print("+2 day",date_tomm)
            return date_tomm
    

    
    # To remove the time stamp from the date index
    def mod_dataset(df1):
        ser = pd.to_datetime(df1.index).to_series()
        df2 = df1.set_index(ser.apply(lambda d : d.date() ))
        return df2
    
    # To predict the value and draw the graph
    def predict_value(x_value):
        scaler = MinMaxScaler(feature_range=(0,1))
        df = pull_data(x_value)
        df_close = df['Close'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(df_close)
        model = load_model('Kera_Stock.h5')
        last_60_days = scaled_data[-60:]
        #Create and empty list
        X_test = []
        # Append the past 60 days
        X_test.append(last_60_days)
        # Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        #Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        # Get the predicted scaled price
        pred_price = model.predict(X_test)
        # Undo the scaling
        pred_price = scaler.inverse_transform(pred_price)
        # Print the last x days
        df_last_x_days = df[-7:]
        #drop the remaning columns
        df_last_x_days_mod = df_last_x_days.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis=1)
        # Find tomorrow's date
        tomm_date = date_weekend()
        
        # Add the new predicted data to the datafrome
        new_row = pd.DataFrame([float(pred_price)], columns = ["Close"], index=[tomm_date])
        df1 = df_last_x_days_mod.append(new_row)
        
        # To remove the time stamp from the date index
        df2 = mod_dataset(df1)
        
        # To show the table in streamlit with the predicted price
        st.dataframe(df2)
                
        # Visualize the data
        curr_stock = df2[:-1]
        pred_stock = df2[-2:]
        plt.figure(figsize=(16,10))
        plt.title('Model')
        plt.xlabel('Date', fontsize= 18)
        plt.ylabel('Close Price USD ($', fontsize= 18)
        plt.plot(curr_stock['Close'])
        plt.plot(pred_stock['Close'])
        plt.legend(['Current prices', 'Prediction'], loc = 'lower right')
        st.pyplot()
        #plt.show()   # To show the Graph in streamlit

    
    app_mode = st.selectbox("Stock",
        ["", "Apple", "Amazon","Tesla"])
    if app_mode == "":
        st.subheader("Select a Stock")
    else:
        run_process(app_mode)
        if st.sidebar.checkbox('Predict Price'):
            st.subheader("Prediction")
            predict_value(app_mode)



if __name__ == "__main__":
    main()