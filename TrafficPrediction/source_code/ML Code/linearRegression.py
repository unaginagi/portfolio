import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Load the data into a DataFrame
data = pd.read_csv('combined_data.csv')

# Prepare the data
X = data.iloc[:, 2:5].values
X = pd.get_dummies(data.drop(['Traffic_Volume'], axis=1)) # One-hot encode categorical variables
y = data['Traffic_Volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split the data into training and testing sets

# Select a model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print('MAE:', mae)

# Make a prediction
new_data = pd.DataFrame({'Time_of_Day': ['2023-03-17 10:00:00'], 'Day_of_Week': ['Thursday'], 'Weather_Condition': ['Clear'], 'Road_Name': ['Main Street'], 'Traffic_Speed': [60.0]})
new_data = pd.get_dummies(new_data)
new_data = new_data.reindex(columns=X.columns, fill_value=0) # Add missing columns
prediction = model.predict(new_data)
print("Predicted traffic volume:", prediction)

#Linear Regression does not work, prediction outcome is too high (1.4 trillion traffic volume)

