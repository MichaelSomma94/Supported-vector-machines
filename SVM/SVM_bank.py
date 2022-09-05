# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 09:56:59 2022

@author: Michael
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import plotly.io as pio

pio.renderers.default = "browser"

data = pd.read_csv("bill_authentication.csv")

data.dropna()

print(data.head())

print(data.shape)

# figure = px.scatter(data_frame = data, x="Variance",
#                     y="Skewness", color="Class")
# figure.show()
#preparing the independent and the dependent variables

X = data.drop('Class', axis=1)
y = data['Class']

#spliting the data in training and test data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.10, random_state=42)

svm_clf = make_pipeline(StandardScaler(), PCA(), SVC(kernel='poly', degree=3)) 

svm_clf.fit(xtrain, ytrain.ravel())
ypred = svm_clf.predict(xtest)

# to access the coefficients later on
classifier = svm_clf.named_steps['svc']
print(svm_clf)
print(svm_clf.score(xtest, ytest))

# get the separating hyperplane
scaler1 = StandardScaler()
scaler1.fit(xtest)
X_test_scaled = scaler1.transform(xtest)
print(X_test_scaled[:, [1, 2]])

pca2 = PCA(n_components=2)
X_test_scaled_reduced = pca2.fit_transform(X_test_scaled)



svm_model = SVC(kernel='poly', degree=3)

classify = svm_model.fit(X_test_scaled[:, [0, 2]], ytest) #X_test_scaled_reduced

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    print ('initial decision function shape; ', np.shape(Z))
    Z = Z.reshape(xx.shape)
    print ('after reshape: ', np.shape(Z))
    out = ax.contourf(xx, yy, Z, **params)
    return out

def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))#,
                          #np.arange(z_min, z_max, h))
    return xx, yy

X0, X1 = X_test_scaled_reduced[:, 0], X_test_scaled_reduced[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(figsize=(12,9))
fig.patch.set_facecolor('white')
cdict1={0:'lime',1:'deeppink'}

Y_tar_list = ytest.tolist()
yl1= [int(target1) for target1 in Y_tar_list]
labels1=yl1
 
labl1={0:'Malignant',1:'Benign'}
marker1={0:'*',1:'d'}
alpha1={0:.8, 1:0.5}

for l1 in np.unique(labels1):
    ix1=np.where(labels1==l1)
    ax.scatter(X0[ix1],X1[ix1], c=cdict1[l1],label=labl1[l1],s=70,marker=marker1[l1],alpha=alpha1[l1])

ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=40, facecolors='none', 
            edgecolors='navy', label='Support Vectors')

plot_contours(ax, classify, xx, yy,cmap='seismic', alpha=0.4)
plt.legend(fontsize=15)

plt.xlabel("1st Principal Component",fontsize=14)
plt.ylabel("2nd Principal Component",fontsize=14)

#plt.savefig('ClassifyMalignant_Benign2D_Decs_FunctG10.png', dpi=300)
plt.show()