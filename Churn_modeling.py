# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 14:27:01 2018

@author: mohana
"""

# standardiser <=> NORMALISER 
# Tte les variables doivent être sur la même échelle.

# Theanos : module de calcul basé sur numpy, utilise le GPU et le CPU à la fois
# tensors flow : module de calul numérique 
# Keras : regroupe les deux et permets de créer les réseau de neurones
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values # Creation de la matrice des variables indépendantes

y = dataset.iloc[:, 13].values
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,  OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])

onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling / NOrmalise
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importation des modules de keras
import keras
from keras.models import Sequential # Permet de créer le classifier
from keras.layers import Dense #permet d'initialiser les poids à des valeurs proches de zéro sans être nulle
from keras.layers import Dropout
#Initialisation

classifier=Sequential()
# Ajout de la couche d'entrée et une couche caché # Fonction d'activation relu <=> redresseur
# le nombre de neurone est un hyper paramètre
classifier.add(Dense(units=4,activation="relu",kernel_initializer="uniform",input_dim=11))
# Ajout des fonctions drop out pour éviter le surapprentissage
classifier.add(Dropout(rate=0.1)) # 10% des neurones vont être désactivé lors de l'apprentissage

#Ajout de la deuxieme couche caché
classifier.add(Dense(units=4,activation="relu",kernel_initializer="uniform"))# plus besoin de input dim car on sait que c'est la couche précédente
classifier.add(Dropout(rate=0.1)) # 10% des neurones vont être désactivé lors de l'apprentissage

#Ajout de la couche de sortie # fonction d'activation <=> sigmoide <=> probabilé # si plusieurs neurones de sortie alors remplacer la fonction d'activation par softmax
classifier.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))

# compilé le réseau # adma gradient stockastique
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"]) # loss = fonction de coût #metrics

#entrainer le réseau de neurones
classifier.fit(X_train,y_train,batch_size=10,epochs=100) # batch_size nombre d'observation par époque

# prediction les donnéessur le test set
y_pred=classifier.predict(X_test) #chaque sortie est la probabilité
# pour construire la matrice de confusion, il faut mettre des 1 et 0
y_pred=(y_pred>0.5) # Pas forcement nécessaire de prendre 50% suivant le cas, on peut prendre moins

# Crocher dans crochet par classifier prend une ligne et [] est une colonne.
# pour retrouver le codage, il faut le faire manuellement
#Pour normaliser, reprendre l'objet standard scaler sc
new_prediction=classifier.predict(sc.transform(np.array([[0.,0,600,0,40,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction>0.5)
# matrice de configusion

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)# les éléments sur la diagonale sont justes. On a environ 85% de prediction juste
# Modèle précis : biais faible
# souvent précis : variance faible

# K Flod cross valdiation
# A partir du jeu training, on le divise en par exemple 100 morceau
# 1er itration avec 9 premiers en entraniement et 1 test
# 2eme iitation avec 19 entrainement + 1 test
# ainsi de suite jusqu'à traiter l'ensemble du jeu de donnéetest
# 
from keras.wrappers.scikit_learn import KerasClassifier
# fait le pont entre keras et scikit learng
from sklearn.model_selection import cross_val_score

# pour construire le classifier 
# comme recréer un reseau de neurone
def build_classifier():
    classifier=Sequential()

    classifier.add(Dense(units=6,activation="relu",
                         kernel_initializer="uniform",input_dim=11))

    classifier.add(Dense(units=6,activation="relu",
                         kernel_initializer="uniform"))

    classifier.add(Dense(units=1,activation="sigmoid",
                         kernel_initializer="uniform"))

    classifier.compile(optimizer="adam",loss="binary_crossentropy",
                       metrics=["accuracy"]) 
    return classifier

#Construit le réseau de neuronne (keras) en scikit learn
classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
#Valeur du cross validation
# CV : strategie de division pour la validation croisée 10 suffisants pour avoir une idée de la variance et biaiss
# n_jobs = nombre de CPU utilisé 

precisions = cross_val_score(estimator=classifier,X=X_train,y=y_train, cv=10)

moyenne = precisions.mean()
ecart_type=precisions.std()
# Drop out: permet d'éviter le sur apprentissage
# Variance très élevé <=> symptone d'un sur apprentissage

# A chauqe entrainement, on va désactiver certain neurone, celà évite une dépendance des neuronnes des un aux autres
#Pour appliquer DROP OUT sur chaque couche

# PArtie 4 GridSearch 
from keras.wrappers.scikit_learn import KerasClassifier
# fait le pont entre keras et scikit learng
from sklearn.model_selection import GridSearchCV #Permet de trouver tout seul les hyper paramètre (nbr de neurone qui vont bien)

# pour construire le classifier 
# comme recréer un reseau de neurone
def build_classifier(optimizer): # Ajout du paramètre optimizer pour le choisir
    classifier=Sequential()

    classifier.add(Dense(units=6,activation="relu",
                         kernel_initializer="uniform",input_dim=11))

    classifier.add(Dense(units=6,activation="relu",
                         kernel_initializer="uniform"))

    classifier.add(Dense(units=1,activation="sigmoid",
                         kernel_initializer="uniform"))

    classifier.compile(optimizer=optimizer,loss="binary_crossentropy",
                       metrics=["accuracy"]) 
    return classifier
#Construit le réseau de neuronne (keras) en scikit learn
classifier = KerasClassifier(build_fn=build_classifier)
# Dictionnaire qui contient les paramètre
parameters={"batch_size":[25,31],
            "epochs":[100,500], # remarque ce nom est directement utlisé dans la fonction
            "optimizer":["adam","rmsprop",]}

grid_search= GridSearchCV(estimator=classifier,
                          param_grid=parameters,
                          scoring="accuracy",
                          cv=10) # Socring métrique de mesure de l'erreur
# Pour cahque combinaison de paramètre, il va diviser le jeu de donnée par 10 et exécuter. 8 combinaison à tester.

# Pour le lancer : 
  
#pour garder les meilleurs paramètres
best_parameters=grid_search.best_params_
best_precision=grid_search.best_score_
# Meilleur optimizer : rmsprop, epochs 500, batch_size 25
# meilleur précision 0.84
classifier.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"]) # loss = fonction de coût #metrics

classifier.fit(X_train,y_train,batch_size=10,epochs=100)