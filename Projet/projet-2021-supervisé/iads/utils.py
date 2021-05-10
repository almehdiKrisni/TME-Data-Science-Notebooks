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
import matplotlib as mpl # Pour les couleurs lors du multi-classe
import math
import random as rd
# ------------------------ 
# Ensemble des fonctions de plot de données

def plot2DSet(data_desc, data_label) :
    # Extraction des exemples de classe -1:
    data_negatifs = data_desc[data_label == -1]
    
    # Extraction des exemples de classe +1:
    data_positifs = data_desc[data_label == +1]
    
    # Affichage de l'ensemble des exemples :
    plt.scatter(data_negatifs[:,0],data_negatifs[:,1],marker='o', color='red') # 'o' pour la classe -1
    plt.scatter(data_positifs[:,0],data_positifs[:,1],marker='x', color='blue') # 'x' pour la classe +1
    
def plot_frontiere_multiclass(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
        cette version reconnaît jusqu'à 4 classes (notée: 0, 1, 2 et 3)
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["skyblue",'darksalmon','#FFDF9E','#B1FB17'],levels=[-1,0,1,2,3,4])
    
def plot_frontiere(desc_set, label_set, classifier, step=3):
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

# calcul de C pour les différentes valeurs de w puis affichage de la courbe correspondante
def calculC(data_set, label_set, allW) :
    # On crée la liste des C
    listC = list()
    
    # On réalise une boucle sur le nombre d'instances de w dans allW
    for w in allW :
        # On crée un perceptron allant réaliser les prédictions avec le w courant
        perc = classif.ClassifierPerceptron(2, 0.1)
        perc.w = w
        
        # On réalise la somme de C
        C = 0
        
        # On itère sur les éléments du data_set
        for i in range(len(data_set)) :
            val = (1 - perc.score(data_set[i]) * label_set[i])
            
            # On vérifie si val > 0 (condition citée en haut)
            if (val > 0) :
                C += val
                
        # On ajoute l'instance de C à la liste de C
        listC.append(C)
    
    # On renvoie la liste des valeurs de C
    return listC

# ------------------------
# code de la validation croisée

def crossval(X, Y, n_iterations, iteration):
    
    # Repartition
    effRep = (len(Y) // n_iterations)
        
    # Création des indices de test
    indexTest = np.asarray([(i + iteration * effRep) for i in range(effRep)])
        
    # Création des indices d'apprentissage
    indexApp = list()
    for i in range(len(X)) :
        if (i not in indexTest) :
            indexApp.append(i)
    indexApp = np.array(indexApp)
        
    # Xapp
    Xapp = X[indexApp]
        
    # Yapp
    Yapp = Y[indexApp]
        
    # Xtest
    Xtest = X[indexTest]
        
    # Ytest
    Ytest = Y[indexTest]
    
    # On retourne les données
    return Xapp, Yapp, Xtest, Ytest

# ------------------------ 
# code de la validation croisée stratifiée
def crossval_strat(X, Y, n_iterations, iteration):
    # Nombre de labels différents
    labels = np.unique(Y)
    
    # On sépare la liste des descriptions et label en nb_labels liste
    # Création de la liste
    liste_classe_desc = []
    liste_classe_label = []
    for l in labels :
        liste_classe_desc.append(X[Y == l])
        liste_classe_label.append(Y[Y == l])
        
    # Création des valeurs de retour
    dim = X.shape[0]
    Xapp = np.array([]).reshape(0, dim)
    Yapp = np.array([])
    Xtest = np.array([]).reshape(0, dim)
    Ytest = np.array([])
    
    # Remplissage des valeurs de retour
    for i in range(len(liste_classe_desc)) :
        # On récupère grace au crossval normal les valeurs de test et d'apprentissage en fonction de l'itération
        XappL, YappL, XtestL, YtestL = crossval(liste_classe_desc[i], liste_classe_label[i], n_iterations, iteration)
        
        # On rassemble les valeurs (stack pour les descriptions, concatenate pour les labels)
        Xapp = np.vstack((Xapp, XappL))
        Yapp = np.concatenate((Yapp, YappL))
        Xtest = np.vstack((Xtest, XtestL))
        Ytest = np.concatenate((Ytest, YtestL))
        
    # Retour des valeurs
    return Xapp, Yapp, Xtest, Ytest
        
        

import copy
import timeit

def calcul_efficacite(classif, X, Y):
    
    # Parametres
    niter = 10 # Nombre d'itération
    perf = [] # Liste des performances des algos

    # Temps de départ
    startTime = timeit.default_timer()
    
    # Boucle principale
    for i in range(niter):
        Xapp,Yapp,Xtest,Ytest = crossval_strat(X, Y, niter, i)
        classif_en_cours = copy.deepcopy(classif)
        for j in range(5):
            classif_en_cours.train(Xapp, Yapp)
        perf.append(classif_en_cours.accuracy(Xtest, Ytest))
        print("Apprentissage ",i+1,":\t"," |Yapp|= ",len(Yapp)," |Ytest|= ",len(Ytest),"\tperf= ",perf[-1])
        
    # Temps de fin
    endTime = timeit.default_timer()

    # On transforme la liste en array numpy pour avoir les fonctions statistiques:
    perf = np.array(perf)

    # Affichage
    print(f'\nTemps mis: --> {endTime - startTime:.5f} secondes')
    print(f'Résultat global:\tmoyenne= {perf.mean():.3f}\técart-type= {perf.std():.3f}')

def centroide(desc) :
    # Renvoie le centroide (sous forme d'array) d'un ensemble d'exemples
    return np.asarray([np.mean(desc[:,i]) for i in range(len(desc[0]))])
        

def normalisation(A) :
    # On renvoie l'array A normalisé
    return np.asarray([((A[i] - A.min(axis=0)) / (A.max(axis=0) - A.min(axis=0))) for i in range(len(A))])

def inertie_cluster(desc) :
    return sum([math.pow(np.linalg.norm(desc[i] - centroide(desc)), 2) for i in range(len(desc))])

def plus_proche(exemple, centroides) :
    # On cherche l'indice du centroide le plus proche
    indice = 0
    for i in range(len(centroides)) :
        if (np.linalg.norm(exemple - centroides[i]) < np.linalg.norm(exemple - centroides[indice])) :
            indice = i
            
    return indice


def affecte_cluster(desc, centroides) :
    matrice = dict()
    for k in range(len(centroides)) : matrice[k] = list()
    for i in range(len(desc)) :
        indice = plus_proche(desc[i], centroides)
        matrice[indice].append(i)
    for k in range(len(centroides)) :
        matrice[k] = np.array(matrice[k])
    return matrice

def nouveaux_centroides(desc, matrice) :
    return [new_centroides.append(centroide(data_norm[matrice[k]])) for k in matrice.keys()]


def inertie_globale(desc, matrice) :
    in_glb = 0
    for i in matrice.keys() :
        in_glb += inertie_cluster(desc[matrice[i]])
        
    return in_glb

def initialisatio(K, desc) :
    selection = []
    print("Hello there")
    while (len(selection) < K) :
        val = rd.randint(0, len(desc) - 1)
        if val not in selection :
            selection.append(val)
    return desc[selection]

def kmoyennes(K, desc, epsilon, iter_max=1000) :
    print("Ici")
    centroides = initialisatio(K, desc)
    affect = affecte_cluster(desc, centroides)
    
    histo_inertie_glb = [inertie_globale(desc, affect)]
    
    for i in range(1, iter_max) :
        centroides = nouveaux_centroides(desc, affect)
        affect = affecte_cluster(desc, centroides)
        # Calcul de la nouvelle inertie globale
        histo_inertie_glb.append(inertie_globale(desc, affect))
        # Affichage de la situation
        difference = mt.sqrt(mt.pow(histo_inertie_glb[i] - histo_inertie_glb[i - 1], 2))
        # Vérification de epsilon
        if (difference < epsilon) :
            return centroides, affect

    return centroides, affect

def affiche_kmoyennes(data, centroides, affectation) :
    # On plot en croix rouges les centroides
    plt.scatter(les_centres[:,0], les_centres[:,1], color='r', marker='x')
    
    # On plot chaque cluster d'une couleur différente
    for i in range(len(centroides)) :
        # On génère une couleur aléatoire
        r = rd.random()
        b = rd.random()
        g = rd.random()
        c = (r, b, g)
        
        # On récupère les points
        data_norm = data[affectation[i]]
        
        # On plot le cluster
        plt.scatter(data_norm[:,0], data_norm[:,1], color=c)