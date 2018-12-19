# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:20:03 2018

@authors: senes, fibs1993
"""

import numpy as np
from math import e
from math import sqrt
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

def relu(x):
    if x < 0:
        return 0
    return x

def der(func):
    if func == identity:
        return lambda x: 1
    if func == sigmoid:
        return lambda x: sigmoid(x) * (1 - sigmoid(x))
    if func == relu:
        return lambda x: 0 if x <= 0 else 1


def my_sum(matrix_list1, matrix_list2):
    return [matrix_list1[i] + matrix_list2[i] for i in range(len(matrix_list1))]

def my_X_scal(scalar, matrix_list):
    return [scalar * matrix for matrix in matrix_list]

def norm(matrix_list):
    norm = 0
    for matrix in matrix_list:
        for row in matrix:
            for elem in row:
                norm += elem**2
    return sqrt(norm)

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
        self.deltas = [np.empty(n_neuron, dtype='float32') for n_neuron in layers]

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
        # lo salviamo come ultimo elemento della lista di matrici che rappresenta il 'gradiente parziale'
        output_prec = np.array([self.act_functs[-2](net) for net in self.deltas[-2]])
        # NB: occhio ai bias
        output_prec = np.append(output_prec, 1)
        # per fare il prodotto vettore colonna per vettore riga bisogna lavorare ancora un po'
        deltas_column_vec = self.deltas[-1].reshape(len(self.deltas[-1]), 1)
        output_prec = output_prec.reshape(1, len(output_prec))
        result[-1] = deltas_column_vec @ output_prec

        # poi per gli hidden layers
        for i in range(-2, -len(self.layers) - 1, -1):  # bisogna scorrere i layers al contrario, fino al secondo, cioè -len(self.layers)
            # NB: i layer sono len(self.layers) + 1 se contiamo anche l'input layer
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

    def fit(self, train_data, train_class, toll, learning_rate, MAX_ITER):
        error = float('inf')
        gradient = [np.ones(layer.shape) for layer in self.layers]
        n_iter = 0
        self.init_weights()

        # NB: i pesi vanno aggiornati solo quando l'errore è troppo grande,
        # quindi appena entro nel while, non alla fine del while
        while(error > toll and n_iter < MAX_ITER and norm(gradient) > 10**(-15) ):
            if n_iter != 0:
                self.layers = my_sum(self.layers, my_X_scal(learning_rate, gradient))
            first = True
            # calcolo del gradiente e dell'errore
            for index, pattern in enumerate(train_data):
                outputNN = self.forward(pattern)
                # out_round = [round(el) for el in outputNN]
                if first:
                    gradient = self.backward(outputNN, train_class[index])
                    error = sum((outputNN - train_class[index]) ** 2)
                    #accuracy_err = sum([abs(el) for el in out_round - train_class[index]])
                    first = False
                else:
                    gradient = my_sum(gradient, self.backward(outputNN, train_class[index]))
                    error += sum((outputNN - train_class[index]) ** 2)
                    #accuracy_err += sum([abs(el) for el in out_round-train_class[index]])
            error = error / len(train_data)
            print(error)
            print(norm(gradient))
            #print(accuracy_err)
            n_iter += 1

        print(n_iter)

        return error

###################-----------PROVA--------###########################

'''     PROVA BACKWARD
NN = NeuralNetwork((2, 2, 2), 2*[sigmoid])
NN.layers = [np.array([[0.15, 0.25, 0.35], [0.2, 0.3, 0.35]]), np.array([[0.4, 0.5, 0.6], [0.45, 0.55, 0.6]])]
out = NN.forward(np.array([0.05, 0.1]))
print(out)
grad = NN.backward(out, np.array([0.8, 0.7]))
print(grad)
'''

''' PROVA MONK

data = np.genfromtxt("Monk1.txt")
target = [np.array(row[0]).astype('float32') for row in data]
train_set = [np.array(row[1:-1]) for row in data]

NN = NeuralNetwork((len(train_set[0]), 3, 3, 1), 3*[sigmoid])
error = NN.fit(train_set, target, 0.001, 0.1, 1000)

data_test = np.genfromtxt("TESTMONK1.txt")
target_test = [np.array(row[0]).astype('float32') for row in data_test]
train_set_test = [np.array(row[1:-1]) for row in data_test]

prediction = NN.predict(train_set_test)
error_test = sum([sum((prediction[i]-target_test[i])**2) for i in range(len(prediction))])


count = 0
for i in range(len(prediction)):
    print(i, prediction[i], target_test[i])
    if prediction[i] != target_test[i]:
        count += 1

print(error_test/len(prediction), count)
'''

''' PROVA TRAINING SET '''
data = np.genfromtxt("ML-CUP18-TR.csv", delimiter=',')[:, 1:]
# normalization of data
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

target = [np.array(row[-2:]).astype('float32') for row in data]
train_set = [np.array(row[:-2]) for row in data]

NN = NeuralNetwork((len(train_set[0]), 50, 50, 2), 3*[relu])
error = NN.fit(train_set, target, 0.0001, 0.05, 500)
print('Errore = ', error)
