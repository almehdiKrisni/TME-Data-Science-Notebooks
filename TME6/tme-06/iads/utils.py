# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2021

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 

def plot2DSet(data_desc, data_label) :
    # Extraction des exemples de classe -1:
    data_negatifs = data_desc[data_label == -1]
    
    # Extraction des exemples de classe +1:
    data_positifs = data_desc[data_label == +1]
    
    # Affichage de l'ensemble des exemples :
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color='red') # 'o' pour la classe -1
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color='blue') # 'x' pour la classe +1
    
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])   
    
    
# ------------------------ 


# genere_dataset_uniform:
def genere_dataset_uniform(p, n, borne_min, borne_max) :
    # Création des valeurs de descriptions
    data_desc = np.random.uniform(borne_min, borne_max, (n * 2, p))
    
    # Création des labels correspondants
    data_label = np.asarray([-1 for i in range(n)] + [+1 for i in range(n)])
    
    # Retour du tuple
    return (data_desc, data_label)
    
# genere_dataset_gaussian:
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points) :
    
    # Création des descriptions négatives
    negative_data_desc = np.random.multivariate_normal(negative_center, negative_sigma, nb_points)
    
    # Création des descriptions positives
    positive_data_desc = np.random.multivariate_normal(positive_center, positive_sigma, nb_points)
    
    # Fusion des descriptions
    data_desc = np.concatenate((negative_data_desc, positive_data_desc), axis=0)
    
    # Création des labels
    data_label = np.asarray([-1 for i in range(nb_points)] + [+1 for i in range(nb_points)])
    
    # Retour du tuple des données
    return (data_desc, data_label)


# ------------------------ 


def create_XOR(n, alpha) :
    # On génère le coin supérieur gauche (+1)
    upper_left_desc = np.random.multivariate_normal(np.array([0,1]), np.array([[alpha,0],[0,alpha]]), n)
    
    # On génère le coin supérieur droit (-1)
    upper_right_desc = np.random.multivariate_normal(np.array([1,1]), np.array([[alpha,0],[0,alpha]]), n)
    
    # On génère le coin inférieur gauche (-1)
    lower_left_desc = np.random.multivariate_normal(np.array([0,0]), np.array([[alpha,0],[0,alpha]]), n)
    
    # On génère le coin inférieur droit (+1)
    lower_right_desc = np.random.multivariate_normal(np.array([1,0]), np.array([[alpha,0],[0,alpha]]), n)
    
    # Création des labels
    data_label = np.asarray([+1 for i in range(n)] + [-1 for i in range(n)] + [-1 for i in range(n)] + [+1 for i in range(n)])
    
    # Retour des descriptions et des labels
    data_desc = np.concatenate((upper_left_desc, upper_right_desc, lower_left_desc, lower_right_desc), axis=0)
    return (data_desc, data_label)


 # ------------------------ 
