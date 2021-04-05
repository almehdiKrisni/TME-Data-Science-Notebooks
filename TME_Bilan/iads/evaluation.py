# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd

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
# ------------------------ 
