# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:20:03 2018

@author: senes
"""

import numpy as np
from math import e
from random import uniform

''' 
FUNZIONI DI ATTIVAZIONE NOTE
'''
def identity(x):
    return x
    
def sign(x):
    if x >= 0:
        return 1
    return -1

def sigmoid(x):
    return 1 / (1 + e**(-x))

def der(func):
    if func == identity:
        return lambda x: 1
    if func == sigmoid:
        return lambda x: sigmoid(x) * (1 - sigmoid(x))

def check_zero(matrix_list):
    for matrix in matrix_list:
            for row in matrix:
                for elem in row:
                    if elem != 0:
                        return False
    return True

'''
Una rete neurale viene rappresentata con una lista di matrici, una per ogni layer diverso 
dall'input layer. Ogni matrice ha una riga per ogni neurone presente nel layer attuale e una colonna
per ogni arco entrante in ciascun nodo e una colonna per il bias, quindi il numero di colonne è uguale
al numero di neuroni del layer precedente più 1. La rete neurale viene inizializzata tramite: 
- una lista di interi, dove l'i-esimo intero rappresenta il numero di neuroni dell'i-esimo layer;
- una lista di stringhe, dove l'i-esima stringa rappresenta la funzione di attivazione da utilizzare
  nell'i-esimo layer (se la stringa non rappresenta una funzione di attivazione nota: NameError) 
'''

class NeuralNetwork:
    def __init__(self, layers, act_functs):
        self.layers = [np.empty((layers[i], layers[i-1] + 1), dtype='float32') for i in range(1, len(layers))]
        self.act_functs = act_functs
        self.deltas = [np.empty(layers[i], dtype='float32') for i in range(len(layers))]
        
    def forward(self, sample):
        # nel forward voglio che deltas rappresenti l'output di ogni neurone
        self.deltas[0] = sample
        input_arr = np.append(sample, 1)
        for j in range(len(self.layers) - 1):   # per ogni hidden layer
            output_arr = self.layers[j] @ input_arr
            self.deltas[j + 1] = output_arr     # ci salviamo i nets
            output_arr = np.array([self.act_functs[j](element) for element in output_arr])
            input_arr = np.append(output_arr, 1)
        # e ora l'output layer
        output_arr = self.layers[-1] @ input_arr
        self.deltas[-1] = output_arr            # ci salviamo i nets
        
        return np.array([self.act_functs[-1](element) for element in output_arr])
        
    def backward(self, outputNN, target_arr):
        # dobbiamo calcolare la 'derivata parziale' per ogni peso, 
        # quindi il risultato ha la stessa struttura dei layers; lo copio tanto poi sovrascrivo
        result = self.layers.copy()   
        
        # prima per l'output layer
        error_arr = target_arr - outputNN
        # sovrascriviamo i net dell'ultimo layer (salvati in self.deltas[-1]) con i delta
        # per farlo ci serve il vettore delle derivate delle funzioni di att calcolate nei net
        derF_arr = np.array([der(self.act_functs[-1])(net) for net in self.deltas[-1]])
        self.deltas[-1] = error_arr * derF_arr
        # il vettore delle derivate parziali relative agli ultimi archi è il prodotto tra il vettore colonna 
        # dei delta e il vettore riga degli output del layer precedente;
        # lo salviamo come ultimo elemento della lista di matrici che rappresenta il 'gradiente' 
        output_prec = np.array([self.act_functs[-2](net) for net in self.deltas[-2]])
        # NB: occhio ai bias
        output_prec = np.append(output_prec, 1)
        # per fare il prodotto vettore colonna per vettore riga bisogna lavorare ancora un po'
        deltas_column_vec = self.deltas[-1].reshape(len(self.deltas[-1]), 1)
        output_prec = output_prec.reshape(1, len(output_prec))
        result[-1] = deltas_column_vec @ output_prec
        
        # poi per gli hidden layers
        for i in range(-2, -len(self.layers) - 1, -1):  # bisogna scorrere i layers al contrario, fino al secondo, cioè -len(self.layers) + 1
            # cambia il modo di calcolare i delta, ma serve sempre il vettore delle derivate ecc..
            derF_arr = np.array([der(self.act_functs[i])(net) for net in self.deltas[i]])
            # e lo dobbiamo moltiplicare con il vettore che in posizione i ha la somma su j di
            # delta_neurone_succ_jesimo * peso_arco_tra_i_e_j
            # NB: i bias non servono!
            weight_matr = self.layers[i + 1][:, :-1]    # cancello la colonna dei bias
            deltas_sum = self.deltas[i + 1].reshape((1, len(self.deltas[i + 1]))) @ weight_matr
            # deltas sum è un vettore riga, voglio che sia un vettore 1D
            deltas_sum = deltas_sum.ravel()
            self.deltas[i] = deltas_sum * derF_arr
            # e nuovamente dobbiamo fare il prodotto vettore riga per vettore colonna
            if i > -len(self.layers):
                output_prec = np.array([self.act_functs[i - 1](net) for net in self.deltas[i - 1]])
            else:   # se il layer precedente è l'input layer non applico le funzioni di attivazione
                output_prec = self.deltas[i-1]
            output_prec = np.append(output_prec, 1)
            output_prec = output_prec.reshape(1, len(output_prec))
            deltas_column_vec = self.deltas[i].reshape(len(self.deltas[i]), 1)
            result[i] = deltas_column_vec @ output_prec
        
        return result
    
    def init_weights(self):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                for k in range(len(self.layers[i][j])):
                    self.layers[i][j][k] = uniform(-0.7, 0.7)
    
    
    def predict(self, data):
        output_arr = []
        len_data = len(data)
        for i in range(len_data):
            outputNN = self.forward(data[i])
            output_arr.append([round(elem) for elem in outputNN])
        
        return output_arr
    
    def fit(self, train_x, train_y, tollerance, learn_rate):
        MAXATTEMPT = 3
        n_attempt = 0
        best_error = 100.0
        best_weights = 0
        MAXITER = 1000
        
        # inizializziamo i pesi diverse volte per trovare i migliori pesi iniziali
        while n_attempt < MAXATTEMPT:
            self.init_weights()
            final_weights = 0
            iter_count = 0
            len_train = len(train_x)
            error_rate = 100.0
            grad = [np.zeros((len(matrix), len(matrix[0])), dtype = 'float32') for matrix in self.layers]
            # ci fermiamo con l'iterazione solo quando l'errore è accettabile 
            # o se si supera il numero massimo di iterazioni
            ############## o se il gradiente è zero
            while error_rate > tollerance and iter_count < MAXITER:
                print(iter_count)
                First = True
                grad = [matr *learn_rate for matr in grad]
                self.layers = [self.layers[i] +  grad[i] for i in range(len(grad))]
                grad = [np.zeros((len(matrix), len(matrix[0])), dtype = 'float32') for matrix in self.layers]
                # per ogni record del TS:
                # calcoliamo l'output
                # aggiorniamo la somma degli errori
                # aggiorniamo il gradiente parziale tramite il backward
                for i in range(len_train):
                    outNN = self.forward(train_x[i]).astype('float32')
                    if First:
                        error_rate = sum((train_y[i] - outNN)**2)
                        First = False
                    else:
                        error_rate += sum((train_y[i] - outNN)**2)
                    
                    gradparz = self.backward(outNN, train_y[i])
                    grad = [gradparz[j] +  grad[j] for j in range(len(grad))] 
                
                # dopo aver iterato su tutto il TS calcoliamo l'errore medio
                #print(error_rate)
                error_rate = error_rate / len_train
                print(error_rate)
                
                iter_count += 1
                ############## se il gradiente = 0 ci fermiamo 
                if check_zero(grad):
                    break
            
            final_weights = self.layers
            # se l'errore medio appena trovato è migliore del best_error aggiorno il best_error e i best_weights
            if error_rate < best_error:
                best_error = error_rate
                best_weights = final_weights
            
            n_attempt += 1
        
        # una volta calcolato il miglior errore e i migliori pesi si assegnano i pesi migliori alla rete
        if best_error != error_rate:
            self.layers = best_weights
        
        return best_error
            


###################-----------PROVA--------###########################
"""NN = NeuralNetwork((2, 2, 2), 3*[sigmoid])
NN.layers = [np.array([[0.15, 0.25, 0.35], [0.2, 0.3, 0.35]]), np.array([[0.4, 0.5, 0.6], [0.45, 0.55, 0.6]])]
out = NN.forward(np.array([0.05, 0.1]))
print(out)
grad = NN.backward(out, np.array([0.8, 0.7]))
print(grad)"""

data_test = np.genfromtxt("TESTMONK1.txt")
target_test = [np.array(row[0]).astype('float32') for row in data_test]
train_set_test = [np.array(row[1:-1]) for row in data_test]

data = np.genfromtxt("Monk1.txt")
target = [np.array(row[0]).astype('float32') for row in data]
train_set = [np.array(row[1:-1]) for row in data]

NN = NeuralNetwork((len(train_set[0]), 3, 3, 1), 3*[sigmoid])
error = NN.fit(train_set, target, 0.001, 0.1)
print(error)

prediction = NN.predict(train_set_test)
error_test = sum([sum((prediction[i]-target_test[i])**2) for i in range(len(prediction))])


count = 0
for i in range(len(prediction)):
    print(i, prediction[i], target_test[i])
    if prediction[i] != target_test[i]:
        count += 1

print(error_test/len(prediction), count)
