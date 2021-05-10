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
import matplotlib as mpl # Pour les couleurs en multi-classe
import math as mt
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
    
def plot2DSetMulticlass(Xmulti,Ymulti):  
    """ ndarray * ndarray -> affichage
    """
    # Extraction de la valeur de chaque classe, soit les différentes valeurs de label
    label = np.unique(Ymulti)
    
    # Sélection d'une liste de couleur pour chacune des classes
    couleurs = list(mpl.colors.cnames.keys())[:len(label)]
    
    # On récupère les descriptions de chaque classe à partir du label
    desc = [Xmulti[Ymulti == l] for l in label]
    
    # On plot chacune des classes
    for i in range(len(label)):
        plt.scatter(desc[i][:,0], desc[i][:,1], color = couleurs[i])
    
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
    dim = X.shape[1]
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

# -----------------------
# fonctions liées à l'algorithme k-means

# Fonction normalisant les données
def normalisation(A) :
    # On renvoie l'array A normalisé
    return np.asarray([((A[i] - A.min(axis=0)) / (A.max(axis=0) - A.min(axis=0))) for i in range(len(A))])

# Fonction retournant la distance vectorielle entre deux points
def dist_vect(p1, p2) :
    # Renvoie la distance vectorielle entre les points p1 et p2
    return np.linalg.norm(p1 - p2)

# Fonction retournant le centroide d'un ensemble de point
def centroide(desc) :
    # Renvoie le centroide (sous forme d'array) d'un ensemble d'exemples (un cluster)
    return np.asarray([np.mean(desc[:,i]) for i in range(len(desc[0]))])

# Fonction retournant l'inertie d'un cluster
def inertie_cluster(desc) :
    # Renvoie la valeur de l'inertie de l'ensemble
    
    # Calcul du centroide des exemples
    centre = centroide(desc)
    
    # Calcul de l'inertie du cluster
    # for i in range(len(desc)) :
    #   print("centre:", centre, "\tExemple:", desc[i], "\tdistance =", mt.pow(np.linalg.norm(centre - desc[i]), 2))
    return sum([mt.pow(np.linalg.norm(desc[i] - centre), 2) for i in range(len(desc))])

# Fonction réalisant la selection des centroides initiaux
def initialisation(K, desc) :
    # On renvoie K exemples tirés aléatoirements
    selection = []
    
    while (len(selection) < K) :
        val = rd.randint(0, len(desc) - 1)
        if val not in selection :
            selection.append(val)
    
    return desc[selection]

# Fonction retournant l'indice du centroide le plus proche d'un point parmi une liste de centroides
def plus_proche(exemple, centroides) :
    # On cherche l'indice du centroide le plus proche
    indice = 0
    
    # On parcourt les differents centroides afin de trouver le plus proche
    for i in range(len(centroides)) :
        if (np.linalg.norm(exemple - centroides[i]) < np.linalg.norm(exemple - centroides[indice])) :
            indice = i
            
    return indice

# Fonction retournant la matrice d'affectation aux clusters des points
def affecte_cluster(desc, centroides) :
    # On renvoie la matrice d'affectation des exemples aux clusters
    
    # Préparation de la matrice
    matrice = dict()
    for k in range(len(centroides)) :
        matrice[k] = list()
        
    # On parcout l'ensemble des exemples et on append au cluster correspondant
    for i in range(len(desc)) :
        indice = plus_proche(desc[i], centroides)
        matrice[indice].append(i)
        
    # On convertit en nparray les données
    for k in range(len(centroides)) :
        matrice[k] = np.array(matrice[k])
        
    # On renvoie la matrice
    return matrice

# Fonction permettant la mise à jour des centroides des clusters
def nouveaux_centroides(desc, matrice) :
    # On renvoie la position des nouveaux centroides
    new_centroides = list()
    
    # On parcourt les points liés à chaque centroide
    for k in matrice.keys() :
        if (matrice[k] != []) :
            liste = matrice[k]
            new_centroides.append(centroide(desc[matrice[k]]))
        
    # On renvoie les nouveaux centroides
    return new_centroides

# Fonction retournant l'inertie globale d'une partition
def inertie_globale(desc, matrice) :
    # On renvoie l'inertie globale d'une base de données
    in_glb = 0

    # On parcourt la matrice d'affectation
    for i in matrice.keys() :
        if (matrice[i] != []) : # On vérifie que le cluster est lié à au moins un point
            in_glb += inertie_cluster(desc[matrice[i]])
        
    return in_glb

# Fonction réalisant l'algorithme des K-moyennes (K-means)
def kmoyennes(K, desc, epsilon=0.01, iter_max=1000) :
    # On renvoie un ensemble de centroides et une matrice d'affectation
    
    # On initialise les K centroides
    centroides = initialisation(K, desc)
    
    # On affecte les données à un cluster
    affect = affecte_cluster(desc, centroides)
    
    # Historique des inerties globales
    # print("Iteration 0\tInertie :", inertie_globale(desc, affect))
    histo_inertie_glb = [inertie_globale(desc, affect)]
    
    for i in range(1, iter_max) :
        # Recalcul des centroides
        centroides = nouveaux_centroides(desc, affect)
        
        # Nouvelle affectation
        affect = affecte_cluster(desc, centroides)
        
        # Calcul de la nouvelle inertie globale
        histo_inertie_glb.append(inertie_globale(desc, affect))
        
        # Affichage de la situation
        difference = mt.sqrt(mt.pow(histo_inertie_glb[i] - histo_inertie_glb[i - 1], 2))
        # print("Iteration", i, "\tInertie :", histo_inertie_glb[i], "\tDifference :", difference)
        
        # Vérification de epsilon
        if (difference < epsilon) :
            # print("\nSortie de l'algorithme - Difference des inerties inférieure à epsilon\n")
            return centroides, affect
        
    # print("\nSortie de l'algortihme - Nombre d'itérations maximal dépassé\n")
    return centroides, affect

# Affiche le résultat graphique de l'algorithme k-moyennes
def affiche_resultat(data, centroides, affectation) : 
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
        
    # On plot en croix rouges les centroides
    centroides = np.array(centroides)
    plt.scatter(centroides[:,0], centroides[:,1], color='r', marker='x')