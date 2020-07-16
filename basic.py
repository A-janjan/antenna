# I recommend you to execute these code with anaconda
# I personally wrote the ipython file first 

#Machine Learning - Multiple Regression

import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O

df=pd.read_csv('antenna.csv')

df=df.dropna()
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')

df.head()

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

    # Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename='meta material antenna'
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

plotPerColumnDistribution(df, 10, 4)

import seaborn as sns
sns.pairplot(df)

#data Correlation
df.corr()

plotCorrelationMatrix(df, 8)

#our dependent and independent variable
X= df[['Xa', 'Ya']]
Y=df['bandwidth']
X,Y = np.array(X), np.array(Y)

# data visulization
sns.jointplot(df['Xa'],df['bandwidth'],kind='hex')
sns.jointplot(df['Ya'],df['bandwidth'],kind='hex')

# 3D plot
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
i=df['Xa']
j=df['Ya']
k=df['bandwidth']
ax.scatter3D(i, j, k, c='blue');

##Multiple Linear Regression
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X , Y)

r_sq=regr.score(X,Y)
print('coefficient of determination:',r_sq,'\n')
if r_sq>0.5:
    print('Multiple Linear Regression is good for this data')
else:
    print('the other methods can be more useful')

#Polynomial Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

## degree = 2
###Transform input data
x_2 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)

###Create a model and fit it
model2=LinearRegression().fit(x_2,Y)

###get results (degree=2)
r_sq=model2.score(x_2,Y)
intercept, coefficients = model2.intercept_,model2.coef_

print('r_squered is : ',r_sq)
print('intercept is : ',intercept)
print('the coefficients is: ',coefficients)

## degree = 3
x_3 = PolynomialFeatures(degree=3, include_bias=False).fit_transform(X)
model3=LinearRegression().fit(x_3,Y)

###get results (degree=3)
r_sq=model3.score(x_3,Y)
intercept, coefficients = model3.intercept_,model3.coef_

print('r_squered is : ',r_sq)
print('intercept is : ',intercept)
print('the coefficients is: ',coefficients)

## degree = 4
x_4 = PolynomialFeatures(degree=4, include_bias=False).fit_transform(X)
model4=LinearRegression().fit(x_4,Y)

###get results (degree=4)
r_sq=model4.score(x_4,Y)
intercept, coefficients = model4.intercept_,model4.coef_

print('r_squered is : ',r_sq)
print('intercept is : ',intercept)
print('the coefficients is: ',coefficients)

# train/test


#80% training_set     /    20% testing_set
tx1=X[:int(0.4*509)]
tx2=X[int(0.6*509):]
train_x=np.concatenate((tx1, tx2))

ty1=Y[:int(0.4*509)]
ty2=Y[int(0.6*509):]
train_y=np.concatenate((ty1, ty2))

test_x=X[int(0.4*509):int(0.6*509)]
test_y=Y[int(0.4*509):int(0.6*509)]

##degree=2 (train/test)

###Transform input data
train_x2 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(train_x)
test_x2 = PolynomialFeatures(degree=2, include_bias=False).fit_transform(test_x)

###Create a model and fit it
n_model2=LinearRegression().fit(train_x2,train_y)

###get results (train/test) (degree=2)
r_sq=n_model2.score(test_x2,test_y)
intercept, coefficients = n_model2.intercept_,model2.coef_

print('r_squered is : ',r_sq)
print('intercept is : ',intercept)
print('the coefficients is: ',coefficients)

##degree=3 (train/test)

###Transform input data
train_x3 = PolynomialFeatures(degree=3, include_bias=False).fit_transform(train_x)
test_x3 = PolynomialFeatures(degree=3, include_bias=False).fit_transform(test_x)

###Create a model and fit it
n_model3=LinearRegression().fit(train_x3,train_y)

###get results (train/test) (degree=3)
r_sq=n_model3.score(test_x3,test_y)
intercept, coefficients = n_model3.intercept_,model3.coef_

print('r_squered is : ',r_sq)
print('intercept is : ',intercept)
print('the coefficients is: ',coefficients)

##degree=4 (train/test)

###Transform input data
train_x4 = PolynomialFeatures(degree=4, include_bias=False).fit_transform(train_x)
test_x4 = PolynomialFeatures(degree=4, include_bias=False).fit_transform(test_x)

###Create a model and fit it
n_model4=LinearRegression().fit(train_x4,train_y)

###get results (train/test) (degree=4)
r_sq=n_model4.score(test_x4,test_y)

intercept, coefficients = n_model4.intercept_,model4.coef_

print('r_squered is : ',r_sq)
print('intercept is : ',intercept)
print('the coefficients is: ',coefficients)


##degree=4 (predict)

k=np.array([[5700, 6482]])
###Transform input data
k = PolynomialFeatures(degree=4, include_bias=False).fit_transform(k)

predicted = n_model4.predict(k)
print(predicted)

xs=np.linspace(2500,9000,1000)
ys=np.linspace(3500,13000,1000)

#predicting the data
mesh = np.array(np.meshgrid(xs, ys))
combinations = mesh.T.reshape(-1, 2)
combinations.shape

###Transform input data
k = PolynomialFeatures(degree=4, include_bias=False).fit_transform(combinations)
predicted = n_model4.predict(k)
predicted.shape

# 3D plot
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
i=np.linspace(2500,9000,1000000)
j=np.linspace(3500,13000,1000000)
k=predicted
ax.scatter3D(i, j, k, c='blue');

y1 = np.asarray(df['bandwidth'], dtype="|S6")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X)

StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
X= scaler.transform(X)

# neural network model

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(2,2,2),max_iter=500)
mlp.fit(X,y1)

# predicting the data
print(mlp.predict([[0.523,0.345]]))

#this is the END, the END, the END