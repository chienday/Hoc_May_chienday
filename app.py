import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import OrdinalEncoder


data = pd.read_csv("student.csv")
data.drop(columns=['StudentID', 'Name', 'Gender'], inplace=True)

list = [['Low', 'Medium', 'High']]
PL = OrdinalEncoder(categories=list)
data['ParentalSupport'] = PL.fit_transform(data[['ParentalSupport']])

# input va output data
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=42)

# Train models
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

lassoreg = Lasso(alpha=0.01)
lassoreg.fit(X_train, Y_train)

mlp_model_neural = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=0)
mlp_model_neural.fit(X_train, Y_train)

# Create Bagging models
#  Bagging1
model_1 = regressor
bagging_model1 = BaggingRegressor(model_1,n_estimators=10,random_state=0)
bagging_model1.fit(X_train,Y_train)
#  Bagging2
model_2=lassoreg
bagging_model2 = BaggingRegressor(model_2,n_estimators=10,random_state=0)
bagging_model2.fit(X_train,Y_train)
#  Bagging3
model_3=mlp_model_neural
bagging_model3 = BaggingRegressor(model_3,n_estimators=10,random_state=0)
bagging_model3.fit(X_train,Y_train)
models=[model_1,model_2,model_3]


# Streamlit interface
st.title("Dự đoán kết quả học tập của học sinh")

# Create a form for user input
with st.form("input_form"):
    a = st.slider("Số phần trăm các lớp học mà học sinh tham gia", 0, 100, 1)
    b = st.number_input("Số giờ mà học sinh dành cho việc học mỗi tuần:", value=0)
    c = st.slider("Điểm học sinh đạt được trong kì trước(thang điểm 100):", 0, 100, 1)
    d = st.number_input("Số lượng ngoại khóa mà học sinh tham gia:", value=0)
    e = st.slider("Mức độ hỗ trợ của phụ huynh đối với học sinh:", 0, 2, 1)

    # Submit button
    submitted = st.form_submit_button("Dự đoán")

# When the user submits the form
if submitted:
    features = np.array([[a, b, c, d, e]])
    
    # Predictions with all models
    linear = regressor.predict(features)[0]
    out1= min(max(linear, 0), 100)
    lasso = lassoreg.predict(features)[0]
    out2= min(max(lasso, 0), 100)
    mlp = mlp_model_neural.predict(features)[0]
    out3= min(max(mlp, 0), 100)
    bagging = [model.predict(features)[0] for model in models]
    final_prediction = sum(bagging) / len(models)
    out4= min(max(final_prediction, 0), 100)

    # Display results
    st.subheader("Kết quả điểm cuối kì:")
    st.write(f"**Linear Regression:** {out1:.2f}")
    st.write(f"**Lasso Regression:** {out2:.2f}")
    st.write(f"**Neural Network (MLP):** {out3:.2f}")
    st.write(f"**Bagging:** {out4:.2f}")
