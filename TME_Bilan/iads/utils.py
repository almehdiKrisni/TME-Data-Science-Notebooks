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
    
    # On itere sur n_iterations
    for k in range(0, n_iterations) :
        
        # Repartition et indice
        rep = 1/n_iterations # Répartition = ppurcentage valeurs envoyées en apprentissage
        effRep = (int)(len(X) * rep)
        
        # Création des indices de test
        indexTest = np.asarray([(i + k * effRep) for i in range(effRep)])
        
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
        
        if (k == iteration) :
            # Si il s'agit de l'itération souhaitée, on renvoie les données
            break
    
    # On retourne les données
    return Xapp, Yapp, Xtest, Ytest

# ------------------------ 
# code de la validation croisée stratifiée

def crossval_strat(X, Y, n_iterations, iteration):
    
    # On itere sur n_iterations
    for k in range(0, n_iterations) :
        
        # Repartition et indice
        midEff = (int)(len(X) / 2) # Marche mieux lorsque la taille des données est paire
        repEff = (int)(midEff / n_iterations)
        
        # Création des index de test
        indexTest1 = np.asarray([i for i in range(repEff * k, repEff * (k + 1))])
        indexTest2 = np.asarray([(i + midEff) for i in range(repEff * k, repEff * (k + 1))])
        
        # Création des index d'apprentissage
        indexApp1 = list()
        indexApp2 = list()
         
        for i in range(0, len(X)) :
            if (i < midEff) :
                if (i not in indexTest1) :
                    indexApp1.append(i)
            else :
                if (i not in indexTest2) :
                    indexApp2.append(i)
                    
        indexApp1 = np.array(indexApp1)
        indexApp2 = np.array(indexApp2)
        
        # Concaténation des indices
        indexApp = np.concatenate((indexApp1, indexApp2))
        indexTest = np.concatenate((indexTest1, indexTest2))
        
        
        # Xapp
        Xapp = X[indexApp]
        
        # Yapp
        Yapp = Y[indexApp]
        
        # Xtest
        Xtest = X[indexTest]
        
        # Ytest
        Ytest = Y[indexTest]
        
        if (k == iteration) :
            # Si il s'agit de l'itération souhaitée, on renvoie les données
            break
    
    # On retourne les données
    return Xapp, Yapp, Xtest, Ytest