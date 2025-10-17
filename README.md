# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING :
```
       import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv("C:\\Users\\krishna\\Downloads\\bmi.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
X = df.drop('Index', axis=1)
y = df['Index']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X_scaled, y)

selected_columns = X.columns[selector.get_support()]
df_final = pd.DataFrame(X_selected, columns=selected_columns)
df_final['Index'] = y.reset_index(drop=True)

df_final.to_csv("Selected_Scaled_BMI.csv", index=False)
```
#OUTPUT:
![Screenshot_17-10-2025_111236_localhost](https://github.com/user-attachments/assets/10b9c220-fb5c-4aca-9da4-4f70eb729643)
![Screenshot_17-10-2025_111250_localhost](https://github.com/user-attachments/assets/ef71f2e4-5b6a-4c62-a1f2-793402f50c33)
![Screenshot_17-10-2025_111311_localhost](https://github.com/user-attachments/assets/7f9ef8bf-2bef-4b91-b228-2ab97535d158)



# RESULT:
 the given data is read and feature scaling,feature selection of the given data is performed successfully and processed dataset is saved as Selected_Scaled_BMI.csv
