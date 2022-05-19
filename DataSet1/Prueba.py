# -*- coding: utf-8 -*-
"""
Created on Wed May 18 21:44:54 2022

@author: rodri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score




simplefilter(action='ignore', category=FutureWarning)

url = 'bank-full.csv'
data = pd.read_csv(url)

#Tratamiento de los datos

data.marital.replace(['married', 'single', 'divorced'], [2, 1, 0], inplace= True)
data.education.replace(['unknown', 'primary', 'secondary', 'tertiary'], [0, 1, 2, 3], inplace= True)
data.default.replace(['no', 'yes'], [0, 1], inplace= True)
data.housing.replace(['no', 'yes'], [0, 1], inplace= True)
data.loan.replace(['no', 'yes'], [0, 1], inplace= True)
data.y.replace(['no', 'yes'], [0, 1], inplace= True)
data.contact.replace(['unknown', 'cellular', 'telephone'], [0, 1, 2], inplace= True)

data.drop(['balance', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'job','poutcome'], axis=1, inplace=True)
data.age.replace(np.nan, 41, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)
data.dropna(axis=0, how='any', inplace=True)

# Partir la data en dos

data_train = data[:22605]
data_test = data[22605:]

x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_train.drop(['y'], 1))
y_test_out = np.array(data_train.y) 


# Regresión Logística




# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entreno el modelo
logreg.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')
#validacion Cruzada
# REGRESIÓN LOGÍSTICA CON VALIDACIÓN CRUZADA

kfold = KFold(n_splits=5)

acc_scores_train_train = []
acc_scores_test_train = []
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

for train, test in kfold.split(x, y):
    logreg.fit(x[train], y[train])
    scores_train_train = logreg.score(x[train], y[train])
    scores_test_train = logreg.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = logreg.predict(x_test_out)

print('*'*50)
print('Regresión Logística Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score1 = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score1}')
# MAQUINA DE SOPORTE VECTORIAL
kfold = KFold(n_splits=5)

acc_scores_train_train = []
acc_scores_test_train = []
svc = SVC(gamma='auto',max_iter = 7600)
for train, test in kfold.split(x, y):
    svc .fit(x[train], y[train])
    scores_train_train = svc .score(x[train], y[train])
    scores_test_train = svc .score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc .predict(x_test_out)
# Seleccionar un modelo
#svc = SVC(gamma='auto')

# Entreno el modelo
#svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score2 = f1_score(y_test_out, y_pred, average=None).mean()
print(f'f1: {f1_score2}')
# ARBOL DE DECISIÓN
kfold = KFold(n_splits=5)

acc_scores_train_train = []
acc_scores_test_train = []
arbol = DecisionTreeClassifier(gamma='auto',max_iter = 7600)
for train, test in kfold.split(x, y):
    arbol.fit(x[train], y[train])
    scores_train_train = arbol.score(x[train], y[train])
    scores_test_train = arbol.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = arbol.predict(x_test_out)
# Seleccionar un modelo
#arbol = DecisionTreeClassifier()

# Entreno el modelo
#arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')

# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Mariz de confución")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score3= f1_score(y_test_out, y_pred, average=None).mean()
print(f'f: {f1_score3}')

# RANDOM FOREST

ranforest = RandomForestClassifier()

# Entrenar el modelo
ranforest.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Random Forest')


# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {ranforest.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {ranforest.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {ranforest.score(x_test_out, y_test_out)}')