import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score,  precision_score,  recall_score, f1_score, mean_squared_error, mean_absolute_error

from scipy.stats import uniform, poisson

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import datetime
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r"D:\year3 c\sem2\ML\datasetproj\train.csv")

df.head()

#dropping
df = df.drop(['id','owner_3_score','RATE_owner_3','CAP_AMOUNT_owner_3','RATE_ID_FOR_industry_type','RATE_ID_FOR_avg_net_deposits','RATE_ID_FOR_funded_last_30','RATE_ID_FOR_location','funded_last_30','RATE_ID_FOR_judgement_lien_amount','INPUT_VALUE_ID_FOR_judgement_lien_amount','RATE_ID_FOR_judgement_lien_percent','judgement_lien_percent','owner_2_score','RATE_owner_2','CAP_AMOUNT_owner_2','PERCENT_OWN_owner_2'],axis=1)
df = df.drop(['INPUT_VALUE_ID_FOR_judgement_lien_time','PERCENT_OWN_owner_3','RATE_ID_FOR_fsr','RATE_ID_FOR_judgement_lien_time','INPUT_VALUE_ID_FOR_tax_lien_amount','RATE_ID_FOR_tax_lien_amount','RATE_ID_FOR_tax_lien_percent','INPUT_VALUE_ID_FOR_tax_lien_count','RATE_ID_FOR_tax_lien_count',],axis=1)
df.drop(df.columns[[0]],axis=1, inplace=True)

#fill null
#using mode
print(df.info())
num_val = ['owner_1_score', 'CAP_AMOUNT_owner_1', 'PERCENT_OWN_owner_1', 'years_in_business','fsr', 'INPUT_VALUE_ID_FOR_num_negative_days','INPUT_VALUE_ID_FOR_num_deposits', 'INPUT_VALUE_ID_FOR_monthly_gross', 'INPUT_VALUE_ID_FOR_average_ledger','INPUT_VALUE_ID_FOR_fc_margin', 'INPUT_VALUE_ID_FOR_tax_lien_percent' ,'INPUT_VALUE_ID_FOR_current_position','INPUT_VALUE_ID_FOR_avg_net_deposits', 'INPUT_VALUE_owner_4','CAP_AMOUNT_owner_4','PERCENT_OWN_owner_4','deal_application_thread_id']
col_vals = []
for i in num_val:
#     print(df[i])
    mean_value = df[i].mean()
    # Fill null values with the mean
    df[i].fillna(mean_value, inplace=True)
    col_vals.append(mean_value)

#using mean
str_val = ['RATE_owner_1','RATE_ID_FOR_years_in_business','location','RATE_ID_FOR_num_negative_days','RATE_ID_FOR_num_deposits','RATE_ID_FOR_monthly_gross','RATE_ID_FOR_average_ledger','RATE_ID_FOR_fc_margin','RATE_ID_FOR_current_position','INPUT_VALUE_ID_FOR_industry_type','RATE_owner_4','completion_status']
column_values = []
for col in str_val:
    mode = df[col].mode()[0]
    df[col].fillna(mode, inplace=True)
    column_values.append(mode)


#visualization1
# histo. for each column indiv.
# for col in df:
#     sns.histplot(df[col], kde = True)
#     plt.show()

# # correlation
# df.corr().style.background_gradient(cmap="Blues")


#encoding
#using labelencoding
catg = ['RATE_ID_FOR_current_position','RATE_owner_1','INPUT_VALUE_ID_FOR_industry_type','RATE_owner_4','completion_status','RATE_ID_FOR_years_in_business','location','RATE_ID_FOR_num_deposits','RATE_ID_FOR_num_negative_days']
for i in catg:
    #print(df[i].unique())
    le = preprocessing.LabelEncoder()
    df[i]=le.fit_transform(df[i])

#using onehot
one_hot_cols = ['RATE_ID_FOR_monthly_gross','RATE_ID_FOR_fc_margin','RATE_ID_FOR_average_ledger']
df = pd.get_dummies(df, columns=one_hot_cols)
# print(df.head())
# df.info()

#visualization2
#histo. btw each 2 columns
# df.hist(figsize= [20,15])
# plt.show()
# plt.tight_layout()

# boxplot for showing outliers
ndf=['owner_1_score', 'CAP_AMOUNT_owner_1', 'PERCENT_OWN_owner_1', 'years_in_business','fsr', 'INPUT_VALUE_ID_FOR_num_negative_days','INPUT_VALUE_ID_FOR_num_deposits', 'INPUT_VALUE_ID_FOR_monthly_gross', 'INPUT_VALUE_ID_FOR_average_ledger','INPUT_VALUE_ID_FOR_fc_margin', 'INPUT_VALUE_ID_FOR_tax_lien_percent' ,'INPUT_VALUE_ID_FOR_current_position','INPUT_VALUE_ID_FOR_avg_net_deposits', 'INPUT_VALUE_owner_4','CAP_AMOUNT_owner_4','PERCENT_OWN_owner_4','deal_application_thread_id']
# for col in df.columns:
#     plt.figure(figsize=(70, 45))
#     sns.boxplot(data=df[col], palette='rainbow', orient='h')
#     plt.title(col)
#     plt.show()
#

for col in ndf:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])



# split data into training and validation set
x = df.drop(['completion_status'], axis=1)
y = df['completion_status']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=46)


#scaling data
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns = x.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns = x.columns)


#regg. models
#ridge
ridge_model = Ridge(alpha=0.99)
ridge_model.fit(x_train, y_train)
y_pred = ridge_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rms = np.sqrt(mse)
print('The root mean square error is:', rms)

#lasso
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(x_train, y_train)
y_pred = lasso_model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

#elasticnet
elasticnet = ElasticNet(alpha=0.01)
elasticnet.fit(x_train, y_train)
y_pred = elasticnet.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)




#Model intialization
LR = LogisticRegression(solver = "liblinear")
DTC = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=1)
nvb = GaussianNB()
ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),learning_rate= 1, n_estimators= 200)



#class. models
#model 1
RFC = RandomForestClassifier()
RFC.fit(x_train,y_train)
output3 = RFC.predict(x_test)
print("RandomForestClassifier:")
print('accuracy score: ',  accuracy_score(y_test,output3))
print("Precision Score : ",precision_score(y_test, output3, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y_test, output3,pos_label='positive',average='macro'))
print('f1 score: ' ,f1_score(y_test,output3,pos_label='positive',average='weighted'))
# cf3 = confusion_matrix(y_test, output3)
cm1 = confusion_matrix(y_test,output3, labels=RFC.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=RFC.classes_)
disp1.plot()


#model 2
DTC.fit(x_train,y_train)
output2 = DTC.predict(x_test)
print("DecisionTreeClassifier:")
print('accuracy score: ',  accuracy_score(y_test,output2))
print("Precision Score : ",precision_score(y_test, output2, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y_test, output2,pos_label='positive',average='micro'))
print('f1 score: ' ,f1_score(y_test,output2,pos_label='positive',average='weighted'))
cm2 = confusion_matrix(y_test,output2, labels=DTC.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=DTC.classes_)
disp1.plot()

#model 3
knn.fit(x_train,y_train)
output1 = knn.predict(x_test)
print("knn:")
print('accuracy score: ',  accuracy_score(y_test,output1))
print("Precision Score : ",precision_score(y_test, output1, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y_test, output1,pos_label='positive',average='micro'))
print('f1 score: ' , f1_score(y_test,output1,pos_label='positive',average='weighted'))
cm3 = confusion_matrix(y_test,output1, labels=knn.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=knn.classes_)
disp1.plot()

#model 4
ada.fit(x_train,y_train)
output5 = ada.predict(x_test)
print("AdaBoostClassifier:")
print('Accuracy score:', (accuracy_score(y_test, output5)))
print("Precision Score : ",precision_score(y_test, output5, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y_test, output5,pos_label='positive',average='micro'))
print('f1 score:' , f1_score(y_test,output5,pos_label='positive',average='weighted'))
cm6 = confusion_matrix(y_test,output5, labels=ada.classes_)
disp5 = ConfusionMatrixDisplay(confusion_matrix=cm6,display_labels=ada.classes_)
disp5.plot()

#model 5
LR.fit(x_train,y_train)
output = LR.predict(x_test)
print("Logistic Regression:")
print('accuracy score: ' , accuracy_score(y_test,output))
print("Precision Score : ",precision_score(y_test, output, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y_test, output,pos_label='positive',average='micro'))
print('f1 score: ' ,f1_score(y_test,output,pos_label='positive',average='weighted'))
cm1 = confusion_matrix(y_test,output, labels=LR.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=LR.classes_)
disp1.plot()

# #model 6
# from sklearn.svm import SVC
# svm_model = SVC(kernel='linear')
# svm_model.fit(x_train, y_train)
# y_pred = svm_model.predict(x_test)
# print('accuracy score: ' , accuracy_score(y_test,y_pred))
# cm4 = confusion_matrix(y_test,y_pred, labels=svm_model.classes_)
# disp1 = ConfusionMatrixDisplay(confusion_matrix=cm4,display_labels=svm_model.classes_)
# disp1.plot()


#best hyp.para. for adaboost using dt estim.
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.5, 1],
    'base_estimator__max_depth': [1, 2, 3]
}
# Create an instance of the AdaBoostClassifier
# clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

# Create a GridSearchCV object with the specified parameters
grid_search = GridSearchCV(ada, param_grid=param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(x_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

#tuning on random forest
n_estimators = [50, 100, 150]
max_depth = [3, 5, 7]
max_features = ["sqrt", "log2"]

param_grid = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "max_features": max_features
}

# Create an instance of the RandomForestClassifier
# clf = RandomForestClassifier()

# Create a GridSearchCV object with the specified parameters
grid_search = GridSearchCV(RFC, param_grid=param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(x_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)




#TESTFILE
##############################################################################################################################
dftest = pd.read_csv(r"D:\year3 c\sem2\ML\datasetproj\train.csv")

#drop
dftest.drop(dftest.columns[[0]], axis=1, inplace=True)
dftest = dftest.drop(['id'], axis=1)

for col in dftest.columns:
    if dftest[col].isnull().sum() / len(dftest) >= 0.8:
        dftest = dftest.drop(col, axis=1)
print("first", dftest.shape[1])


#mean
i=0
for col in num_val:
    dftest[col].fillna(col_vals[i], inplace=True)
    i+=1
#mode
j=0
for col in str_val:
    dftest[col].fillna(column_values[j], inplace=True)
    j+=1

print(dftest.info())


#encoding
#using labelencoding
catg = ['RATE_ID_FOR_tax_lien_count','RATE_ID_FOR_tax_lien_percent','RATE_ID_FOR_tax_lien_amount','RATE_ID_FOR_judgement_lien_amount' ,'RATE_ID_FOR_judgement_lien_percent','funded_last_30','RATE_ID_FOR_location','RATE_ID_FOR_judgement_lien_amount','RATE_ID_FOR_current_position','RATE_owner_1','INPUT_VALUE_ID_FOR_industry_type','RATE_owner_4','completion_status','RATE_ID_FOR_years_in_business','location','RATE_ID_FOR_num_deposits','RATE_ID_FOR_num_negative_days']
for i in catg:
    #print(df[i].unique())
    le = preprocessing.LabelEncoder()
    dftest[i]=le.fit_transform(dftest[i])
# for col in dftest.columns:
#     if dftest[col].dtype == 'object':  # check if column is string
#         if len(dftest[col].unique()) > 2:  # check if there are more than 2 unique values
#             encoder = LabelEncoder()
#             dftest[col] = encoder.fit_transform(dftest[col])
#         else:
#
#             encoder = OneHotEncoder()
#             temp = pd.DataFrame(encoder.fit_transform(dftest[[col]]).toarray())
#             temp.columns = [col+'_'+str(i) for i in range(len(temp.columns))]
#             dftest = pd.concat([dftest, temp], axis=1)
#             dftest.drop(col, axis=1, inplace=True)
#using onehot
one_hot_cols = ['RATE_ID_FOR_monthly_gross','RATE_ID_FOR_fc_margin','RATE_ID_FOR_average_ledger']
dftest = pd.get_dummies(dftest, columns=one_hot_cols)
# boxplot for showing outliers
print("testtttt")
# dftest.info()
# for col in dftest.columns:
#     plt.figure(figsize=(70, 45))
#     sns.boxplot(data=dftest[col], palette='rainbow', orient='h')
#     plt.title(col)
#     plt.show()

#dropping columns with many outliers
# df = df.drop(['years_in_business', 'RATE_ID_FOR_num_negative_days', 'INPUT_VALUE_ID_FOR_average_ledger'], axis=1)
# outliers_vals = [ 'PERCENT_OWN_owner_1', 'years_in_business','INPUT_VALUE_ID_FOR_num_deposits', 'INPUT_VALUE_ID_FOR_monthly_gross', 'INPUT_VALUE_ID_FOR_average_ledger', 'INPUT_VALUE_ID_FOR_fc_margin', 'INPUT_VALUE_ID_FOR_tax_lien_percent' ,'INPUT_VALUE_ID_FOR_current_position', 'INPUT_VALUE_ID_FOR_avg_net_deposits', 'INPUT_VALUE_owner_4', 'CAP_AMOUNT_owner_4', 'PERCENT_OWN_owner_4', 'deal_application_thread_id']
# outl_vals = []
# for i in outliers_vals:
#     # Calculate the median value of the column
#     median_value = df[i].median()
#     # Replace outliers with the median value
#     df[i] = np.where((df[i] < df[i].quantile(0.05)) | (df[i] > df[i].quantile(0.95)), median_value, df[i])
#     outl_vals.append(median_value)
# dft = pd.DataFrame(outl_vals)
# # dft = [float(x) for x in n]
# dfall = pd.concat([dft, df], axis = 1)
# dfall.head()
for col in ndf:
    Q1 = dftest[col].quantile(0.25)
    Q3 = dftest[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dftest[col] = np.where(dftest[col] < lower_bound, lower_bound, dftest[col])
    dftest[col] = np.where(dftest[col] > upper_bound, upper_bound, dftest[col])
#scaling
# transform DataFrame
dftest.info()
dftest= pd.DataFrame(scaler.transform(dftest), columns=dftest.columns)




#features and target
x = dftest.drop(['completion_status'], axis=1)
y = dftest['completion_status']

#Model intialization
LR = LogisticRegression(solver = "liblinear")
DTC = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=1)
nvb = GaussianNB()
ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),learning_rate= 1, n_estimators= 200)


print("TEST:")
#class. models
#model 1
RFC = RandomForestClassifier()
RFC.transform(x,y)
output3 = RFC.predict(x)
print("RandomForestClassifier:")
print('accuracy score: ',  accuracy_score(y,output3))
print("Precision Score : ",precision_score(y, output3, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y, output3,pos_label='positive',average='macro'))
print('f1 score: ' ,f1_score(y,output3,pos_label='positive',average='weighted'))
# cf3 = confusion_matrix(y_test, output3)
cm1 = confusion_matrix(y,output3, labels=RFC.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=RFC.classes_)
disp1.plot()


#model 2
DTC.transform(x,y)
output2 = DTC.predict(x)
print("DecisionTreeClassifier:")
print('accuracy score: ',  accuracy_score(y,output2))
print("Precision Score : ",precision_score(y, output2, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y, output2,pos_label='positive',average='micro'))
print('f1 score: ' ,f1_score(y,output2,pos_label='positive',average='weighted'))
cm2 = confusion_matrix(y,output2, labels=DTC.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=DTC.classes_)
disp1.plot()

#model 3
knn.transform(x,y)
output1 = knn.predict(x)
print("knn:")
print('accuracy score: ',  accuracy_score(y,output1))
print("Precision Score : ",precision_score(y, output1, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y, output1,pos_label='positive',average='micro'))
print('f1 score: ' , f1_score(y,output1,pos_label='positive',average='weighted'))
cm3 = confusion_matrix(y,output1, labels=knn.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=knn.classes_)
disp1.plot()

#model 4
ada.transform(x,y)
output5 = ada.predict(x)
print("AdaBoostClassifier:")
print('Accuracy score:', (accuracy_score(y, output5)))
print("Precision Score : ",precision_score(y, output5, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y, output5,pos_label='positive',average='micro'))
print('f1 score:' , f1_score(y,output5,pos_label='positive',average='weighted'))
cm6 = confusion_matrix(y,output5, labels=ada.classes_)
disp5 = ConfusionMatrixDisplay(confusion_matrix=cm6,display_labels=ada.classes_)
disp5.plot()

#model 5
LR.transform(x,y)
output = LR.predict(x)
print("Logistic Regression:")
print('accuracy score: ' ,accuracy_score(y,output))
print("Precision Score : ",precision_score(y, output, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y, output,pos_label='positive',average='micro'))
print('f1 score: ' ,f1_score(y,output,pos_label='positive',average='weighted'))
cm1 = confusion_matrix(y,output, labels=LR.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=LR.classes_)
disp1.plot()

# #model 6
# from sklearn.svm import SVC
# svm_model = SVC(kernel='linear')
# svm_model.fit(x_train, y_train)
# y_pred = svm_model.predict(x_test)
# print('accuracy score: ' , accuracy_score(y_test,y_pred))
# cm4 = confusion_matrix(y_test,y_pred, labels=svm_model.classes_)
# disp1 = ConfusionMatrixDisplay(confusion_matrix=cm4,display_labels=svm_model.classes_)
# disp1.plot()


#best hyp.para. for adaboost using dt estim.
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.5, 1],
    'base_estimator__max_depth': [1, 2, 3]
}
# Create an instance of the AdaBoostClassifier
# clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

# Create a GridSearchCV object with the specified parameters
grid_search = GridSearchCV(ada, param_grid=param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(x, y)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

#tuning on random forest
n_estimators = [50, 100, 150]
max_depth = [3, 5, 7]
max_features = ["sqrt", "log2"]

param_grid = {
    "n_estimators": n_estimators,
    "max_depth": max_depth,
    "max_features": max_features
}

# Create an instance of the RandomForestClassifier
# clf = RandomForestClassifier()

# Create a GridSearchCV object with the specified parameters
grid_search = GridSearchCV(RFC, param_grid=param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(x,y)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)







