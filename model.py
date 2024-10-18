import os
os.getcwd()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

shop = pd.read_csv("online_shoppers_intention.csv")
shop.head()

shop.describe()

shop.shape

shop.isnull().sum()

def convert_month(month_str):
  try:
    return pd.to_datetime(month_str, format='%b').month
  except ValueError:
    try:
      return pd.to_datetime(month_str, format='%B').month
    except ValueError:
      try:
        return int(month_str)
      except ValueError:
        return None
shop['Month'] = shop['Month'].apply(convert_month)

shope = shop.drop(['Administrative', 'Informational', 'ProductRelated_Duration'], axis = 1)

shope.head()

shope['ProductRelated'].value_counts()

shope['Month'].value_counts()

shope['OperatingSystems'].value_counts()

print(shope.columns)
print(shop.columns)

shope['VisitorType'].value_counts()

shope['Browser'].value_counts()

shope['Region'].value_counts()

visitor = pd.get_dummies(shope['VisitorType'])

X = pd.concat([shope, visitor], axis=1)
X.head()

X.columns

y = X['Revenue']
X_new = X.drop(['Revenue', 'Month', 'VisitorType'], axis = 1)

X_new['Weekend'] = np.asarray(X_new['Weekend']).astype(np.float32)

y = np.asarray(y).astype(np.float32)

y.shape

X_new.shape

model = Sequential()
model.add(Dense(units=18,activation="relu"))
model.add(Dense(units=32,activation="relu"))
model.add(Dense(units=16,activation="relu"))

model.add(Dense(units=2,activation="softmax"))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_new,y,epochs=100, validation_split=0.1)

model.save("my_model.h5")

# Load the model and save it as a .pkl file
import pickle  # Import the pickle module
from tensorflow.keras.models import load_model

# Load the model from the .h5 file (optional, if you want to save as .pkl)
loaded_model = load_model("my_model.h5")

# Save the loaded model using pickle
with open("my_model.pkl", "wb") as file:
    pickle.dump(loaded_model, file)

print("Model saved as my_model.pkl")



