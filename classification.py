import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline

data=pd.read_csv('mnist.csv')

data1=pd.read_csv('mnist_train.csv')

type(data)

data

data.head()


data.info()

data.iloc[:,3]

a=data.iloc[3,1:].values
a=a.reshape(28,28).astype('uint')
plt.imshow(a)

b=data.iloc[2,1:].values
b=b.reshape(28,28).astype('uint')
plt.imshow(b)
