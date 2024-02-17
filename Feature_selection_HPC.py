import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("Data/DSB_BDK_trainingset.csv", sep=";")
random_state=42

feature_information = pd.DataFrame(columns=['Navn','Cat/Num','Datatype','Min', 'Max', 'Mean', 'Std', 'Notes'])
# Easily extract needed value from the decribe function with integer-conversion 
def extract_from_describe(described, descriptor):
    value = described[descriptor]
    if value.is_integer():
        return int(value)
    return value

# Insert rows in dataframe
cat_var = []
for col in data.columns:
    if data[col].dtypes == 'O':
        df = pd.DataFrame({'Navn': col, 'Cat/Num': 'Categorical'}, index=[0])
        feature_information = pd.concat([feature_information, df], ignore_index=True)
        cat_var += [col]
    else:
        col_data = data[col].describe()
        min1 = extract_from_describe(col_data, 'min')
        max1 = extract_from_describe(col_data, 'max')
        mean1 = extract_from_describe(col_data, 'mean')
        std1 = extract_from_describe(col_data, 'std')
        datatype = 'float'
        if np.array_equal(data[col], data[col].astype(int)):
            datatype = 'int'
        df = pd.DataFrame({'Navn': col, 'Cat/Num': 'Numerical','Datatype': datatype,'Min':min1
                           , 'Max':max1, 'Mean': mean1, 'Std': std1}, index=[0])
        feature_information = pd.concat([feature_information, df], ignore_index=True)

# Only looking at Kystbanen
# We exclude the "Categorical" columns for now
data_20 = data[data["visualiseringskode"] == 20]
data_20 = data_20.drop(cat_var,axis=1)

# Divide into input and target.
# Divide into train and test set. 
X = data_20.iloc[:,1:]
y = data_20.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = random_state)


regressor = RandomForestRegressor(n_estimators=100, random_state=random_state, oob_score=True, max_depth=5)
ffs = SequentialFeatureSelector(regressor, k_features="best", forward=True, n_jobs=-1)
ffs.fit(X_train,y_train)
features = list(ffs.k_feature_names_)
regressor.fit(X_train[features],y_train)
y_pred=regressor.predict(X_test[features])

acc = np.mean(y_pred == y_test)
print(acc)