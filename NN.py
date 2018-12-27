# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:20:03 2018

@authors: senes, fibs1993
"""

import numpy as np
from numpy.random import shuffle
import matplotlib.pyplot as plt
from math import e
from math import sqrt
from math import tanh
from random import uniform

class Error(Exception):
    pass

class InputError(Error):
    pass

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

def my_tanh(x):
    return tanh(x)

def der(func):
    if func == identity:
        return lambda x: 1
    if func == sigmoid:
        return lambda x: sigmoid(x) * (1 - sigmoid(x))
    if func == relu:
        return lambda x: 0 if x <= 0 else 1
    if func == my_tanh:
        return lambda x: 1 - tanh(x)**2


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

def MSE(predicted_list, real_list):
    error_list = [sum((predicted_list[i] - real_list[i])**2) for i in range(len(predicted_list))]
    return sum(error_list) / len(error_list)
'''
Una rete neurale viene rappresentata con una lista di matrici, una per ogni layer diverso
dall'input layer. Ogni matrice ha una riga per ogni neurone presente nel layer attuale e una colonna
per ogni arco entrante in ciascun nodo e una colonna per il bias, quindi il numero di colonne è uguale
al numero di neuroni del layer precedente più 1. Queste matrici vengono inizializzate nel metodo fit, che 
serve ad allenare la rete, cioè nel primo momento in cui si conoscono le dimensioni dell'input e del'output.
La rete neurale viene inizializzata tramite:
- hidden_layer: una lista di interi, dove l'i-esimo intero rappresenta il numero di neuroni dell'i-esimo hidden layer;
- act_functs: una lista di stringhe, dove l'i-esima stringa rappresenta la funzione di attivazione da utilizzare
  nell'i-esimo layer (se la stringa non rappresenta una funzione di attivazione nota: NameError)
NB: le funzioni di attivazione sono una in più degli hidden layer
- toll: float, minimo valore accettabile come errore
- learning_rate: float
- max_iter: int, numero massimo di volte in cui si scorre il dataset per calcolare il gradiente della funzione da minimizzare
- Lambda: float, rappresenta il parametro utilizzato per la regolarizzazione
- n_init: int, numero di volte in cui vengono inizializzati i pesi nel metodo fit 
'''

class NeuralNetwork:
    def __init__(self, hidden_layers, act_functs, toll=0.01, learning_rate=0.04, max_iter=200, Lambda=0.000001, n_init=5):
        if len(act_functs) != len(hidden_layers) + 1:
            raise InputError()
        self.hidden_layers = hidden_layers
        self.weights = []
        self.act_functs = act_functs
        self.deltas = []
        self.toll = toll
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.Lambda = Lambda
        self.n_init = n_init

    def forward(self, sample):
        # nel forward voglio che deltas rappresenti l'output di ogni neurone
        self.deltas[0] = sample
        input_arr = np.append(sample, 1)
        for j in range(len(self.weights) - 1):   # per ogni hidden layer
            output_arr = self.weights[j] @ input_arr
            self.deltas[j + 1] = output_arr     # ci salviamo i nets
            output_arr = np.array([self.act_functs[j](element) for element in output_arr])
            input_arr = np.append(output_arr, 1)
        # e ora l'output layer
        output_arr = self.weights[-1] @ input_arr
        self.deltas[-1] = output_arr            # ci salviamo i nets

        return np.array([self.act_functs[-1](element) for element in output_arr])

    def backward(self, outputNN, target_arr):
        # dobbiamo calcolare la 'derivata parziale' per ogni peso,
        # quindi il risultato ha la stessa struttura dei layers; lo copio tanto poi sovrascrivo
        result = self.weights.copy()

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
        for i in range(-2, -len(self.weights) - 1, -1):  # bisogna scorrere i layers al contrario, fino al secondo, cioè -len(self.weights)
            # NB: i layer sono len(self.weights) + 1 se contiamo anche l'input layer
            # cambia il modo di calcolare i delta, ma serve sempre il vettore delle derivate ecc..
            derF_arr = np.array([der(self.act_functs[i])(net) for net in self.deltas[i]])
            # e lo dobbiamo moltiplicare con il vettore che in posizione i ha la somma su j di
            # delta_neurone_succ_jesimo * peso_arco_tra_i_e_j
            # NB: i bias non servono!
            weight_matr = self.weights[i + 1][:, :-1]    # cancello la colonna dei bias
            deltas_sum = self.deltas[i + 1].reshape((1, len(self.deltas[i + 1]))) @ weight_matr
            # deltas sum è un vettore riga, voglio che sia un vettore 1D
            deltas_sum = deltas_sum.ravel()
            self.deltas[i] = deltas_sum * derF_arr
            # e nuovamente dobbiamo fare il prodotto vettore riga per vettore colonna
            if i > -len(self.weights):
                output_prec = np.array([self.act_functs[i - 1](net) for net in self.deltas[i - 1]])
            else:   # se il layer precedente è l'input layer non applico le funzioni di attivazione
                output_prec = self.deltas[i-1]
            output_prec = np.append(output_prec, 1)
            output_prec = output_prec.reshape(1, len(output_prec))
            deltas_column_vec = self.deltas[i].reshape(len(self.deltas[i]), 1)
            result[i] = deltas_column_vec @ output_prec

        return result

    def init_weights(self):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] = uniform(-0.7, 0.7)

    def fit(self, train_data, train_class):
        # creiamo la lista di matrici dei pesi, ora che sappiamo le dimensioni dell'input e dell'output
        # creiamo anche i delta, ora che sappiamo il numero di neuroni di ogni layer
        layers_list = [len(train_data[0])] + list(self.hidden_layers) + [len(train_class[0])]
        self.weights = [np.empty((layers_list[i], layers_list[i-1] + 1), dtype='float32') for i in range(1, len(layers_list))]
        self.deltas = [np.empty(n_neuron, dtype='float32') for n_neuron in layers_list]

        # ora inizia l'algoritmo
        min_error = float('inf')
        best_weights = self.weights
        # i pesi vanno inizializzati più volte
        # ogni volta che li inizializziamo facciamo ripartire l'algoritmo vero e proprio
        # memorizziamo l'errore minimo di ogni tentativo e i pesi migliori
        for n_initialization in range(self.n_init):
            print('inizializzazione numero ', n_initialization + 1)
            error = float('inf')
            gradient = [np.ones(layer.shape) for layer in self.weights]
            n_iter = 0
            self.init_weights()

            # NB: i pesi vanno aggiornati solo quando l'errore è troppo grande,
            # quindi appena entro nel while, non alla fine del while
            while(error > self.toll and n_iter < self.max_iter and norm(gradient) > 10**(-5) ):
                if n_iter != 0:
                    #print(norm(self.weights))
                    self.weights = my_sum(self.weights, my_X_scal(self.learning_rate, gradient))
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
                # regolarizzazione
                if self.Lambda != 0:
                    gradient = my_sum(gradient, my_X_scal(-self.Lambda, self.weights))
                
                error = error / len(train_data)
                #print(error)
                #print(norm(gradient))
                #print(accuracy_err)
                n_iter += 1

            if error < min_error:
                min_error = error
                best_weights = self.weights 

        self.weights = best_weights
        return min_error

    
    def predict(self, data):
        output_arr = []
        len_data = len(data)
        for i in range(len_data):
            outputNN = self.forward(data[i])
            output_arr.append([elem for elem in outputNN])

        return output_arr

    def k_fold_cv(self, data, k=3):
        # calcoliamo la lunghezza di ogni divisione del dataset
        divided_data_size = len(data) // k
        # np.split divide il dataset a seconda degli indici che gli passiamo nella lista (secondo parametro)
        # quindi list_subdata è una lista di un np.ndarray bidimensionali (attributi in colonna, record in riga)
        list_subdata = np.split(data, [i * divided_data_size for i in range(1, k)])
        error_list = []
        for i in range(k):
            # splitting in test and train
            # il test set è semplicemente l'i-esimo elemento della lista delle porzioni del dataset
            test_data = list_subdata[i]
            # il training set è la lista dei record presenti in tutti gli altri elementi della lista delle porzioni
            train_data = []
            for j in range(k):
                if j != i:
                    for row in list_subdata[j]:
                        train_data.append(row)
            # splitting in train attributes, train target, test attr and test target
            train_x = [np.array(row[:-2]) for row in train_data]
            train_y = [np.array(row[-2:]) for row in train_data]
            test_x = [np.array(row[:-2]) for row in test_data]
            test_y = [np.array(row[-2:]) for row in test_data]
            # fit the neural network
            train_error = self.fit(train_x, train_y)
            # calculate test error
            test_predict = self.predict(test_x)
            mean_squared_error = MSE(test_predict, test_y)
            error_list.append((train_error, mean_squared_error))
        
        return error_list

    def MonteCarlo_cv(self, data, n_fit=5, test_percentage=0.7):
        error_list = []
        for _ in range(n_fit):
            shuffle(data)
            # splitting in test and train
            n_train = round(len(data) * test_percentage)
            train_data = data[:n_train, :]
            test_data = data[n_train:, :]
            # splitting in train attributes, train target, test attr and test target
            train_x = [np.array(row[:-2]) for row in train_data]
            train_y = [np.array(row[-2:]) for row in train_data]
            test_x = [np.array(row[:-2]) for row in test_data]
            test_y = [np.array(row[-2:]) for row in test_data]
            # fit the neural network
            train_error = self.fit(train_x, train_y)
            # calculate test error
            test_predict = self.predict(test_x)
            mean_squared_error = MSE(test_predict, test_y)
            error_list.append((train_error, mean_squared_error))
        
        return error_list



###################-----------PROVA--------###########################

'''     PROVA BACKWARD
NN = NeuralNetwork([2], 1*[sigmoid])
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

NN = NeuralNetwork((3, 3), 3*[my_tanh])
error = NN.fit(train_set, target)

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
# eliminiamo la colonna dell'indice
data = np.genfromtxt("ML-CUP18-TR.csv", delimiter=',')[:, 1:]
# normalization of data
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
# splitting in test and train, after we shuffle the dataset
shuffle(data)
test_percentage = 0.7
n_train = round(len(data) * test_percentage)
train_data = data[:n_train, :]
test_data = data[n_train:, :]
# splitting in train attributes, train target, test attr and test target
train_x = [np.array(row[:-2]) for row in train_data]
train_y = [np.array(row[-2:]) for row in train_data]
test_x = [np.array(row[:-2]) for row in test_data]
test_y = [np.array(row[-2:]) for row in test_data]
# prova con parametri 'casuali'

NN = NeuralNetwork( [10, 10], 2 * [my_tanh] + [identity],  learning_rate=0.0001, Lambda=0.5 )
train_error = NN.fit(train_x, train_y)
train_predict = NN.predict(train_x)
test_predict = NN.predict(test_x)
test_error = MSE(test_predict, test_y)
print(train_error)
print(test_error)
# plot dei risultati
# training
plt.scatter([point[0] for point in train_y], [point[1] for point in train_y], c='b', alpha=0.05)
plt.scatter([point[0] for point in train_predict], [point[1] for point in train_predict], c='r', alpha=0.5)
# test
plt.scatter([point[0] for point in test_y], [point[1] for point in test_y], c='k', alpha=0.05)
plt.scatter([point[0] for point in test_predict], [point[1] for point in test_predict], c='y', alpha=0.5)
plt.show()
'''
# creating train error list and test error list, in function of n_neurons and plotting results
train_error_list = []
test_error_list = []
n_neuron_list = range(2, 15)
for n_neuron in n_neuron_list:
    NN = NeuralNetwork((n_neuron), 2*[sigmoid])
    train_error = NN.fit(train_x, train_y)
    print(train_error)
    train_error_list.append(train_error)
    test_predict = NN.predict(test_x)
    error_list = [sum((test_predict[i] - test_y[i])**2) for i in range(len(test_predict))]
    test_error = sum(error_list) / len(error_list)
    print(test_error)
    test_error_list.append(test_error)
plt.plot(n_neuron_list, train_error_list)
plt.plot(n_neuron_list, test_error_list)
plt.show()

## creating train error list and test error list, in function of n_iteration and plotting results
train_error_list = []
test_error_list = []
n_iteration_list = range(1, 200, 10)
NN = NeuralNetwork((10, 10), 3*[sigmoid])
for n_iteration in n_iteration_list:
    train_error = NN.fit(train_x, train_y, n_iteration)
    print(train_error)
    train_error_list.append(train_error)
    test_predict = NN.predict(test_x)
    test_error = MSE(test_predict, test_y)
    print(test_error)
    test_error_list.append(test_error)
plt.plot(n_iteration_list, train_error_list)
plt.plot(n_iteration_list, test_error_list)
plt.show()
'''
