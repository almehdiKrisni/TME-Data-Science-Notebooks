# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2021

# Import de packages externes
import numpy as np
import pandas as pd
import math

# ---------------------------
# code de la classe Classifier

class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.nbDimensions = input_dimension
        #raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """ 
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        
    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # On crée la valeur d'indice de précision
        accuracy_value = 0
        
        # On vérifie si les prédictions et les valeurs brutes concordent
        for i in range(len(label_set)) :
            if self.predict(desc_set[i]) == label_set[i] :
                accuracy_value += 1
             
        # On retourne l'indice de précision
        return accuracy_value / len(label_set)
    
    
# ---------------------------
# code de la classe Lineaire Random

class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.nbDimensions = input_dimension
        self.w = np.asarray([np.random.uniform() for i in range(self.nbDimensions)])
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Pas d'apprentissage pour ce classifieur")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # Création de la valeur du score de prédiction
        prediction_score = 0.0
        
        # On effectue le calcul entre les vecteurs x et w
        for i in range(x.size):
            prediction_score += np.vdot(self.w[i],x[i])
            
        # Retour du score de prédiction
        return prediction_score
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        
        # Création de la valeur de prediction
        prediction = self.score(x)
        
        # On retourne la prédiction appropriée
        if (prediction < 0) :
            return -1
        else :
            return 1   
        
        
# ---------------------------
# code de la classe KNN

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    #TODO: A Compléter
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        # On crée le tableau des distances entre les descriptions et x
        tab_distance = np.asarray([mt.sqrt(mt.pow(x[0] - self.desc_set[i][0], 2) + mt.pow(x[1] - self.desc_set[i][1], 2)) for i in range(100)])
                          
        # On obtient le tableau des ordres en fonction de la distance (min à max)
        ordre_distance = np.argsort(tab_distance)

        # On renvoie le nombre de voisins +1 parmi les k plus proches voisins
        nbv = 0
        for i in range(self.k) :
            for j in range(len(ordre_distance)) :
                if (ordre_distance[j] == i) :
                    if (self.label_set[ordre_distance[j]] > 0) :
                        nbv += 1
                        
        # On retourne la proportion
        return nbv / self.k
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        if self.score(x) >= 0.5 :
            return 1
        else :
            return -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set
        
        
 # ---------------------------
# code de la classe Perceptron

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension,learning_rate, history=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        self.w = np.zeros(input_dimension)
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.history = history
        if (history) :
            self.allw = list()
            self.allw.append(self.w)
        
    def train(self, desc_set, label_set, nitermax=10):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
            Hypothèse: nitermax >= 1
        """
        
        # On sauvegarde le W initial
        WStart = self.w
        
        # On effectue les itérations sur les points x
        for i in range(len(label_set) * nitermax) :
            index = np.random.randint(0, len(label_set) - 1) # On choisit un index au hasard
            desc = desc_set[index]
            
            if (self.score(desc) * label_set[index] <= 1) :
                # On prend en compte l'erreur et on actualise le vecteur w
                self.w = self.w + (self.learning_rate * desc * label_set[index])
                if (self.history) :
                    self.allw.append(self.w)
            
        # raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
       
        # Retour du score de prédiction
        return (self.w @ x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        # Retour de la prédiction
        if (self.score(x) < 0) :
            return -1
        else :
            return 1
        
        
 # ---------------------------
# code de la classe Perceptron Kernel

class ClassifierPerceptronKernel(Classifier):
    """ Perceptron utilisant un kernel
    """
    #TODO: Classe à Compléter
    def __init__(self, input_dimension, learning_rate, noyau):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : 
                - noyau : Kernel à utiliser
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
        
        
# ------------------------ 
# code de la classe ADALINE Analytique

class ClassifierADALINE2(classif.Classifier):
    """ Perceptron de ADALINE
    """
    #TODO: Classe à Compléter
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.w = np.zeros(input_dimension)
        self.input_dimension = input_dimension
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        
        '''
        # On crée la transposée de X et on réalise la multiplication de trX et X
        trX = desc_set.transpose()
        trX_X = np.dot(trX, desc_set)
        
        # On réalise la multiplication de trX et Y
        trX_Y = np.dot(trX, label_set)
        
        # Inverse de la matrice trX_X
        inv_trX_X = np.linalg.inv(trX_X)
        
        # Multiplication des deux matrices
        self.w = np.dot(trX_Y, inv_trX_X)
        '''
        
        # Utilisation de np.linalg.solve
        trX = desc_set.transpose() # Création de la transposee de desc_set
        trX_X = np.dot(trX, desc_set) # Produit factoriel de la transposee de desc_set et desc_set (élem. gauche)
        trX_Y = np.dot(trX, label_set) # Produit factoriel entre desc_set et label_set (élem. droit)
        
        x = np.array([1,1]) # Matrice [x, y]
        a = trX_X * x
        self.w = np.linalg.solve(a, trX_Y)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # Retour du score
        return (self.w @ x)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        # Retour de la prediction
        if (self.score(x) < 0) :
            return -1
        else :
            return 1
