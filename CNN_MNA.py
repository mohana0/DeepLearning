# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 18:48:24 2018

@author: mohana
"""
# Imagine tableau à 3 dimension
# Ajouter l'information de à quoi correspond l'image on le fiat via le nom car on ne peut pas juste ajouter un tableau.
# une meilleur solution consiste à utiliser keras, Pour celà il faut une bonne structure de donnée 
# dans le dossier, on sépare ce qui est chien et ce qui est chat
#  Partie 1 : construction du réseau de neuronnes à convoluion

from keras.models import Sequential # Amortisage possible avec un graphe aussi mais on a tjs une séquence de couche donc sequential
from keras.layers import Convolution2D # image <=> objet en 2D (video en 3d<=> temps)
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense # pour créer le ANN

# Initialiser le CNN
classifier = Sequential() # reseau de neurone initialisé
#Ajout de la couche de convolution 
# constitué de X features map ~ sous partie à reconnaitre, il fuat choisir le nombre de feature detector à utiliser, ici filters=32 
# multiple de 2, 32 pour commencer est le standard en général
#  si on ajoute une couche, il faut multiplier par deux

# kernel size = taille de la feature, 3 ou [3,3] si pas caré
# stride : de combien de pixel on bouge le features detectore 
# input shape : taille de l'image sur laquelle on va appliquer le feature detector, on va les forcer à adapter le même format
# ex 128, 128, x avec x=1 si niveau de gris et 3 si RGB.
# fonction d'activation Relu <=> remplace valeur négative par 0 pour ajouter de la non linéarité
classifier.add(Convolution2D(filters=32,kernel_size=3,strides=1,
               input_shape=(256,256,3),activation="relu")) # fonction d'activation est la fonction redresseur "relu"
# 64.64 trop petit à 200 pixels plus précis

# pooling :
# sur la feature map, prends le max de petit carré 2x2, on a une pooled feature map plus petite
# opération réalisé sur toute les features map
# permet de simplifier le modèle sans perte d'information Car on garde le maximum or c'est le maximum qui permet de détecter les particularités d'un feature
# poolsize : taille de la petite matrice, 2x2 est un bon compromi
# stride pas spécifié car par defaut reprends la valeur de la petite matrice
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Ajout d'une couche de convolution
classifier.add(Convolution2D(filters=32,kernel_size=3,strides=1,
              # input_shape=(64,64,3),#pas la peine de repréciser ici, le réseau sait quel taille d'image il va avoir en entrée car après le pooling la taille a changer
               activation="relu"))
# Ajout de la couche de pooling qui va avec.

classifier.add(MaxPooling2D(pool_size=(2,2)))

# Ajout d'une couche de convolution, on multiplie par deux à chaque deux coucle ? le nombre de featuresmap (filter)
classifier.add(Convolution2D(filters=64,kernel_size=3,strides=1,
              # input_shape=(64,64,3),#pas la peine de repréciser ici, le réseau sait quel taille d'image il va avoir en entrée car après le pooling la taille a changer
               activation="relu"))
# Ajout de la couche de pooling qui va avec.

classifier.add(MaxPooling2D(pool_size=(2,2)))
# Ajout d'une couche de convolution, on multiplie par deux à chaque deux coucle ? le nombre de featuresmap (filter)
classifier.add(Convolution2D(filters=64,kernel_size=3,strides=1,
              # input_shape=(64,64,3),#pas la peine de repréciser ici, le réseau sait quel taille d'image il va avoir en entrée car après le pooling la taille a changer
               activation="relu"))
# Ajout de la couche de pooling qui va avec.

classifier.add(MaxPooling2D(pool_size=(2,2)))
# flattening
# réécrit la pooled feature map sous forme d'un vecteur colonne
# on ne perds pas la structure de l'image, car dans la convolution, on a extrait la valeur élevé, dans pooling on garde aussi la valuer élevé
# si on fait le flattening dès le début, on aurait des neurones avec en entrée des pixels sans aucune information particulière
# la convolution permet d'extraire la structure de l'image, le pooling permet de réduire la taille de cette information et enfin le flattening permet de la réoordonner pour être cimpatible avec un réseau de neuronne

classifier.add(Flatten()) # sans arguments

# ajout du ANN completement connecté
# couche caché = motié de neuronne entrée/sortie
# puissance de 2 marche assez bien
classifier.add(Dense(units=128,activation="relu"))
classifier.add(Dense(units=128,activation="relu"))
classifier.add(Dense(units=128,activation="relu")) # ♦On ajoute 3 couches pour avoir 
# couche de sortie
classifier.add(Dropout(0.3)) # Chaque neurone à 30% de chance d'êttre désactivé lors d'une étpae d'apprentissage, ce qui permet de limiter le sur apprentissage
classifier.add(Dense(units=1,activation="sigmoid")) # softmax si plusieurs neurones de sortie, ici juste deux classes
# compilation
# choix de l'algo de gradient
# fonction de coût : binaire
# metric poru calculer l'erreur 
classifier.compile(optimizer="adam",loss="binary_crossentropy",
                   metrics=["accuracy"])

# Problème de sur entraitenment 
# ImageDataGenerator
# va créer des nouvelles images en transformants les images existantes
# ex : translation, rotation, mirroir, 
# pour l'apprentissage ud réseau de neurone, ca fera des images différentes.
# va permettre de réduire le surentrainement en augmentant la base d'apprentissage
# Augmente le JDD en ajoutant des transformation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255, # tt les valeurs des pixels entre 0 et 1
        shear_range=0.2,# transvection ~ cisaillement de l'image
        zoom_range=0.2,# zoom
        horizontal_flip=True)  # mirroir horizontal (on peut faire vertical aussis suivant les cas)

test_datagen = ImageDataGenerator(rescale=1./255)#On va juste changer l'échelle sur le jeu de test

training_set = train_datagen.flow_from_directory( # va créer les images
        'dataset\\training_set',
        target_size=(150,150), # taille des nouvelles images
        batch_size=32,# sur l'algodu gradient par lot, on divise les observations par lot (au nombre de 32)
        class_mode='binary') #2 classes => OK
# résultat found 80000 images belonging to 2 classes

test_set = test_datagen.flow_from_directory(# va créer les images
        'dataset\\test_set',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

# fit generator va se charger d'entrainer et tester le réseau de neurone
classifier.fit_generator(
        training_set,
        steps_per_epoch=250, # 8000 image/ 32 lot
        epochs=75, # Augmente le nbr d epoque
        validation_data=test_set,
        validation_steps=63) # 2000 image/32 lots
 # avec 1 couches de convoltuoin
#training accuracy =0.8574
# test accuracy= 0.76
 # avec 2 couches de convoltuoin
 #training accuracy =0.88
# test accuracy= 0.82
 # agrandir la couche des images, mettre plus de couches de convolution

# pour faire une prédiction
import numpy as np
from keras.preprocessing import image

test_image = image.load_img("dataset\\single_prediction\\cat_or_dog_3.jpg",target_size=(64,64)) # charge l'image et règle la taille de sortie pour être égale à celui du réseau de neurone
test_image = image.img_to_array(test_image)
test_image =np.expand_dims(test_image, axis=0) #conv2d à une dimension en plus,il faut donc ajouter une dimensioncar on peut avoir plusieur groupe avec plusieur image dans chacun des groupes
result=classifier.predict(test_image)
training_set.class_indices # permet de voir à quel indice correponds quelle classe
if result[0][0]==1:
    prediction="chien"
else:
    prediction="chat"