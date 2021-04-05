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
import math as mt
import copy

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
        # On crée le distance de x au éléments de desc_set
        distances_x = []
        index = []
        for k in range(len(self.desc_set)) :
            distances_x.append(self.distance(x, self.desc_set[k]))
            index.append(k)
            
        # On crée un nouveau tableau contenant la distance à x et l'index pour chaque descriptions et l'ordonne sur la dist.
        new_tab = list(zip(distances_x, index))
        new_tab.sort(key = lambda i: i[0])
        
        # On parcourt les k plus proches voisins de x et on vérifie le nombre de voisins à +1
        nbv = 0
        for i in range(0, self.k) :
            if (self.label_set[new_tab[i][1]] > 0) :
                nbv = nbv + 1
                
        # On renvoie la proportion de voisins à +1
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
        
    def distance(self, x, desc) :
        return mt.sqrt((desc[0] - x[0]) * (desc[0] - x[0]) + (desc[1] - x[1]) * (desc[1] - x[1]))
        
        
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
        
        # On crée un ordre random pour parcourir desc_set
        ordre = np.asarray([i for i in range(len(label_set))])
        np.random.shuffle(ordre)
        
        # On effectue les itérations sur les points x
        for index in ordre :
            desc = desc_set[index]
            
            if (self.score(desc) * label_set[index] <= 0) :
                # On prend en compte l'erreur et on actualise le vecteur w
                self.w = self.w + (self.learning_rate * desc * label_set[index])
                if (self.history) :
                    self.allw.append(self.w)
    
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
        
    def getW(self) :
        return self.w
    
# -------------------------
# code de la classe PerceptronBiais

class ClassifierPerceptronBiais(Classifier):
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
        
        # On crée un ordre random pour parcourir desc_set
        ordre = np.asarray([i for i in range(len(label_set))])
        np.random.shuffle(ordre)
        
        # On effectue les itérations sur les points x
        for index in ordre :
            desc = desc_set[index]
            
            if (self.score(desc) * label_set[index] <= 1) :
                # On prend en compte l'erreur et on actualise le vecteur w
                self.w = self.w + (self.learning_rate * desc * label_set[index])
                if (self.history) :
                    self.allw.append(self.w)
    
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
        
    def getW(self) :
        return self.w
        
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
# code de la classe pour le classifieur ADALINE

class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    #TODO: Classe à Compléter
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
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
        self.learning_rate = learning_rate
        self.history = history
        if (history) :
            self.allw = [self.w]
        self.niter_max = niter_max
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """     
        # On initialise un w random et on régle la variable wPast
        self.w = np.random.uniform(-1, 1, (1, 2))
        wPast = self.w
        
        # On effectue la boucle
        for k in range(self.niter_max) :
            
            # On tire une valeur au hasard
            i = np.random.randint(0, len(desc_set) - 1)
            
            # Calcul du gradient
            # On sépare les calculs pour plus de lisibilité
            trans = desc_set[i].reshape(2, 1)
            temp = np.dot(self.w, desc_set[i]) - label_set[i]
            grad = np.dot(trans, temp)
            
            # Recalcul de la valeur de w
            self.w = self.w - (self.learning_rate * grad)
            
            if (self.history) :
                self.allw.append(self.w)
            
            # Test de convergence
            wPast = self.w
            
        # Fin de la méthode
        return
    
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
        
    def getW(self) :
        return self.w
        
# ------------------------ 
# code de la classe ADALINE Analytique

class ClassifierADALINE2(Classifier):
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
        
        x = np.array([1] * self.input_dimension) # Matrice [x, y]
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
        
    def getW(self) :
        return self.w

# --------------------------
class ClassifierMultiOAA(Classifier):
    """ 
    """
    
    def __init__(self, classifieur):
        """ Constructeur de Classifier
            Argument:
                - classif : le classifieur utilisé
                - liste_classif : la liste de classifieurs utilisés pour classer les données de chaque classe
        """
        self.classif = classifieur
        self.liste_classif = []
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # On obtient le nombre de classes des données
        nb_classes = len(np.unique(label_set))
        
        # On réalise une copie du classifieur self.classif pour chaque classe différente
        self.liste_classif = [copy.deepcopy(self.classif) for i in range(nb_classes)] # On réalise une copie du classifieur
        
        # On crée des listes de labels afin de différencier chaque classe du reste (len(sous_label) = nb_classes)
        sous_label = [[] for i in range(nb_classes)]
        
        # On remplit les listes de label
        for label in label_set :
            
            # On remplit toutes les listes en fonction de l'appartenance à une classe précise
            for i in range(nb_classes) :
                
                # Si la valeur concorde, cela veut dire que l'exemple lié à ce label fait partie de cette classe donc +1
                if (label == i) :
                    sous_label[i].append(1)
                    
                # Sinon, il n'en fait pas partie donc -1
                else:
                    sous_label[i].append(-1)
        
        # On entraine les classifieurs de la liste en fonction des listes de labels
        for i in range(nb_classes):
            self.liste_classif[i].train(desc_set, sous_label[i])
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """ 
        return [classif.score(x) for classif in self.liste_classif] # On obient le score de x pour chaque classifieur
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        predictions = self.score(x)
        return predictions.index(max(predictions))