from cProfile import label
import csv
import pandas as pd
import numpy as np
import random
from sklearn import svm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from natsort import index_natsorted

# from DE import DonorVector


df = pd.read_csv('Train.csv')
df

# print(df)
df_test =pd.read_csv('Test.csv')
df_test



Train = df.drop(columns=["Class"])
Train.head()
Train.shape
# Train.keys()

Test_x = df_test.drop(columns=["Class"])
Test_x.head()
Test_x.shape

Train_Class = df["Class"]
Train_Class.head()
Train_Class.shape

Test_y = df_test["Class"]
Test_y.head()
Test_y.shape

df_columns = []
i = 0
while i <50:
    feature_range = random.randrange(1,100)
    # feature_range
    # print(feature_range)
    df1 = Train.loc[i]
# df_rows = Train.sample()
    df_columns.append(df1.sample( n=feature_range ))
    # df_columns
    # print(df_columns)
    i+=1
    
# df2 = df_columns
# print("---rezvan---")
# print(df_columns)
df_NaN = pd.DataFrame(df_columns)
# df_train = df_train.reset_index()
df_NaN

df_exit = df_NaN.replace(np.nan, 0)
# df_train.sort_index(ascending=False)
df_exit

df_temp = Train * 0

df_train = df_temp + df_exit

df_train = df_train.replace(np.nan, 0)

df_train

# df_clean = df_train.replace([np.inf, -np.inf, np.nan_to_num(0)], np.nan, inplace=True)



model = svm.SVC(kernel='linear', C=1, gamma=1)

model.fit(df_train,Train_Class)

model.score(df_train,Train_Class)

predicted = model.predict(Test_x)
predicted

results = confusion_matrix(Test_y, predicted)
print(results)
print ('Accuracy Score is(target)',accuracy_score(Test_y, predicted))
print ('Classification Report : ')
print (classification_report(Test_y, predicted))

plt.scatter(Test_y, predicted)

print("---------Donor Vector--------")
i = 0
V = []
F = 1.5
while i <50:
       
    
    R1 = random.randrange(0,49)
    R2 = random.randrange(0,49)
    R3 = random.randrange(0,49)
    if R1 == R2:
        R2 = random.randrange(0,49)
    if R2 == R3:
        R3 = random.randrange(0,49)
    if R3 == R1:
        R3 = random.randrange(0,49)
    # print(R1,R2,R3)
    
    V.append(df_train.loc[R1] + F*(df_train.loc[R2] - df_train.loc[R3]))
    # print(V)
    i+=1
df_v = pd.DataFrame(V) 
# df_v.sort_index(ascending=False)
# df_v.sort_values(by=0 ,axis=1)
# df_v.apply(lambda x: x.sort_values().values)
df_v
# df_v.to_csv("df_v.csv")

# df_v ===> Donor vector
# df_train ===> target vector

# TrialVector
print("---Trial Vector---")

Pc= 0.8
g = 1
T = []


# for j in df:
#     if r > Pc and j != g:
#         T = df_train.loc[j]
         
#     elif r <= Pc or j == g:
#         T = df_v.loc[j]

j = 0
while j < 50:
    r = random.SystemRandom().uniform(0, 1)
    # print("r = " , r) 
    if r > Pc and j != g:
        T.append(df_train.loc[j])
    elif r <= Pc or j == g:
        T.append(df_v.loc[j])
    j+=1

df_T = pd.DataFrame(T)
df_T
    
model = svm.SVC(kernel='linear', C=1, gamma=1)

model.fit(df_T,Train_Class)

model.score(df_T,Train_Class)

predicted_T = model.predict(Test_x)
predicted_T

results = confusion_matrix(Test_y, predicted_T)
print(results)
print ('Accuracy Score is(Trial)',accuracy_score(Test_y, predicted_T))
print ('Classification Report : ')
print (classification_report(Test_y, predicted_T))



# new_pop = []

# if accuracy_score(Test_y, predicted_T) > accuracy_score(Test_y, predicted):
#     new_pop.append(df_T)
#     print("bozorg")
# elif accuracy_score(Test_y, predicted_T) <= accuracy_score(Test_y, predicted):
#     new_pop.append(df_v)
#     print("kochik")

# new_pop

# New_poplolation = pd.DataFrame(new_pop)
# New_poplolation