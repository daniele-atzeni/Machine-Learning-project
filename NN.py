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
from copy import deepcopy

class Error(Exception):
    pass

class InputError(Error):
    def __init__(self, message):
        self.message = message

class UntrainedError(Error):
    def __init__(self, message):
        self.message = message

'''
FUNZIONI DI ATTIVAZIONE NOTE
'''
def identity(x):
    return x

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

# creiamo un dizionario che associa nome della funzione come stringa alla funzione stessa
#  cosi quando creiamo la rete neurale in input gli diamo una lista di stringhe di funzioni di attivazione 
dict_funct = {'sigmoid' : sigmoid, 'tanh': my_tanh, 'relu' : relu}


def my_sum(matrix_list1, matrix_list2):
    return [matrix_list1[i] + matrix_list2[i] for i in range(len(matrix_list1))]

def my_prod_per_scal(scalar, matrix_list):
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
    def __init__(self, hidden_layers, act_functs, toll=0.1, learning_rate=0.0005, alpha = 0, minibatch_size=None, max_epochs=200, Lambda=0.001, min_increasing_score = 0.0001, n_init=5, classification=False):
        if len(act_functs) != len(hidden_layers):
            raise InputError('Numero funzioni attivazione != Numero hidden layers')
        self.hidden_layers = hidden_layers
        # inizializzazione della lista di funzioni di att partendo dalla lista di stringhe usando dict_funct
        # mettendo come ultima funzione di attivazione sigmoid se dobbiamo classificare
        global dict_funct
        try:
            act_functs = [dict_funct[string] for string in act_functs]
        except:
            raise InputError('Una delle funzioni non è stata riconosciuta, lista funzioni valide: ' + str([func for func in dict_funct.keys()]))
        if classification:
            act_functs += [sigmoid]
        else:
            act_functs += [identity]
        self.act_functs = act_functs
        self.weights = None
        self.deltas = None
        self.toll = toll
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.minibatch_size = minibatch_size
        self.max_epochs = max_epochs
        self.Lambda = Lambda
        self.min_increasing_score = min_increasing_score
        self.n_init = n_init

    def _forward(self, sample):
    # ritorna un np.array con gli output
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

    def _backward(self, outputNN, target_arr):
    # ritorna una lista di matrici (stessa forma dei layer) con il 'gradiente parziale'
        # dobbiamo calcolare la 'derivata parziale' per ogni peso,
        # quindi il risultato ha la stessa struttura dei layers; lo copio tanto poi sovrascrivo
        result = deepcopy(self.weights)

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

    def _init_weights(self):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] = uniform(-0.7, 0.7)

    def _update_weights(self, gradient):
        # regularization
        if self.Lambda != 0:
            gradient = my_sum(gradient, my_prod_per_scal(-self.Lambda, self.weights))
        # update weights
        self.weights = my_sum(self.weights, my_prod_per_scal(self.learning_rate, gradient))
        return 

    def fit(self, train_x, train_y):
        # creiamo la lista di matrici dei pesi, ora che sappiamo le dimensioni dell'input e dell'output
        # creiamo anche i delta, ora che sappiamo il numero di neuroni di ogni layer
        layers_list = [len(train_x[0])] + list(self.hidden_layers) + [len(train_y[0])]
        self.weights = [np.empty((layers_list[i], layers_list[i-1] + 1), dtype='float32') for i in range(1, len(layers_list))]
        self.deltas = [np.empty(n_neuron, dtype='float32') for n_neuron in layers_list]
        # se minibatch_size = 0 ---> versione batch dell'algoritmo, quindi minibatch_size = len(train_x)
        if self.minibatch_size == None:
            self.minibatch_size = len(train_x)

        # ora inizia l'algoritmo
        min_error = float('inf')
        # i pesi vanno inizializzati più volte
        # ogni volta che li inizializziamo facciamo ripartire l'algoritmo vero e proprio
        # memorizziamo l'errore minimo di ogni tentativo e i pesi migliori
        for n in range(self.n_init):
            print('inizializzazione', n + 1)
            curr_error = float('inf')
            count_not_decreasing = 0
            gradient = [np.zeros_like(layer) for layer in self.weights]
            self._init_weights()

            for n_epochs in range(self.max_epochs):
                # calcolo del gradiente, sommando tutti i risultati di ogni backprop
                for index, pattern in enumerate(train_x):
                    outputNN = self._forward(pattern)
                    gradient = my_sum(gradient, self._backward(outputNN, train_y[index]))
                    # dopo minibatch_size passi aggiorniamo i pesi e reinizializziamo il gradiente
                    # NB: la regolarizzazione viene fatta in update_weights_and_gradient
                    if index != 0 and index % self.minibatch_size == 0:
                        self._update_weights(gradient)
                        # reset the gradient, to 0 if no momentum(alpha = 0)
                        # to alpha times the old gradient otherwise
                        gradient = my_prod_per_scal(self.alpha, gradient)

                # dopo aver visto tutti i pattern bisogna nuovamente aggiornare i pesi
                self._update_weights(gradient)
                gradient = my_prod_per_scal(self.alpha, gradient)
                # calcolo errori
                prev_error = curr_error
                curr_error = self.score(train_x, train_y)
                print(curr_error)

                # controlli per uscire dal ciclo:
                # se l'errore non decrementa per 5 volte di fila usciamo (occhio a questa condizione, la usiamo solo 
                # dopo aver fatto un po' di iterazioni, diciamo 10, perché all'inizio è troppo instabile);
                # se la rete va già sufficientemente bene usciamo
                if prev_error - curr_error < self.min_increasing_score and n_epochs > 10:
                    count_not_decreasing += 1
                else:
                    count_not_decreasing = 0
                if curr_error < self.toll:# or count_not_decreasing >= 10:
                    break

            # alla fine dell'allenamento, se abbiamo ottenuto risultati migliori aggiorniamo min_error e best_weights
            if curr_error < min_error:
                min_error = curr_error
                best_weights = deepcopy(self.weights)

        self.weights = best_weights
        return
    
    def predict(self, data):
    # ritorna una lista di np.array con gli output per ogni pattern
        # errore se la rete non è stata fittata --> non si conosce il numero di input e output
        if not self.weights:
            raise UntrainedError('La rete deve prima essere allenata!')
        
        output_arr = []
        len_data = len(data)
        for i in range(len_data):
            outputNN = self._forward(data[i])
            output_arr.append([elem for elem in outputNN])
        return output_arr

    def score(self, data_x, data_y):
    # ritorna l'accuracy oppure il mean squared error
        # errore se la rete non è stata fittata --> non si conosce il numero di input e output
        if not self.weights:
            raise UntrainedError('La rete deve prima essere allenata!')

        predicted_y = self.predict(data_x)
        # se dobbiamo classificare lo score è l'accuracy
        if self.act_functs[-1] == sigmoid:
            # arrotondiamo i valori ottenuti, ottenendo così una lista di array con 1 o 0
            predicted_classes = [round(prediction) for prediction in predicted_y]
            n_missclass = 0
            for index, predicted_array in enumerate(predicted_classes):
                if predicted_array[0] != data_y[index][0]:
                    n_missclass += 1
            return n_missclass / len(data_y)
        else:
            # NB predicted_y è una lista di array, così come data_y
            error_list = [sum((prediction - data_y[index])**2) for index, prediction in enumerate(predicted_y)]
            return sum(error_list) / len(error_list)

    def k_fold_cv(self, data, k=5):
    # ritorna una lista di score (MSE/accuracy), uno per ogni tentativo
        # errore se la rete non è stata fittata --> non si conosce il numero di input e output
        if not self.weights:
            raise UntrainedError('La rete deve prima essere allenata!')

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
            self.fit(train_x, train_y)
            # calculate test error and append it to the result
            test_error = self.score(test_x, test_y)
            error_list.append(test_error)
        
        return error_list

    def MonteCarlo_cv(self, data, n_fit=5, test_percentage=0.7):
    # ritorna una lista di score, come k-fold CV
        # errore se la rete non è stata fittata --> non si conosce il numero di input e output
        if not self.weights:
            raise UntrainedError('La rete deve prima essere allenata!')

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
            self.fit(train_x, train_y)
            # calculate test error
            test_error = self.score(test_x, test_y)
            error_list.append(test_error)
        
        return error_list



###################-----------PROVA--------###########################

''' PROVA MONK

data = np.genfromtxt("Monk1.txt")
train_y = [np.array(row[0]).astype('float32') for row in data]
train_x = [np.array(row[1:-1]) for row in data]

NN = NeuralNetwork((3, 3), 2*['tanh'], classification=True)
NN.fit(train_x, train_y)
train_error = NN.score(train_x, train_y)

data_test = np.genfromtxt("TESTMONK1.txt")
test_y = [np.array(row[0]).astype('float32') for row in data_test]
test_x = [np.array(row[1:-1]) for row in data_test]

error_test = NN.score(test_x, test_y)
print(error_test)
'''

''' PROVA TRAINING SET '''
# eliminiamo la colonna dell'indice
data = np.genfromtxt("ML-CUP18-TR.csv", delimiter=',')[:, 1:]
# splitting in test and train, after we shuffle the dataset
shuffle(data)
test_percentage = 0.7
n_train = round(len(data) * test_percentage)
train_data = data[:n_train, :]
test_data = data[n_train:, :]
# normalization of data
mean_arr = np.mean(train_data, axis=0)
std_arr = np.std(train_data, axis=0)
train_data = (data - mean_arr) / std_arr
# splitting in train and validation
test_percentage = 0.7
n_train = round(len(data) * test_percentage)
train_data = data[:n_train, :]
val_data = data[n_train:, :]
# splitting in train attributes, train target, test attr and test target
train_x = [np.array(row[:-2]) for row in train_data]
train_y = [np.array(row[-2:]) for row in train_data]
val_x = [np.array(row[:-2]) for row in val_data]
val_y = [np.array(row[-2:]) for row in val_data]

# prova con parametri 'casuali'
NN = NeuralNetwork( 3 * [20], 3 * ['tanh'], alpha=0.3, n_init=1, learning_rate=0.0002, minibatch_size=32)
NN.fit(train_x, train_y)
train_error = NN.score(train_x, train_y)
val_error = NN.score(val_x, val_y)
print(train_error)
print(val_error)
# plot dei risultati
train_predict = NN.predict(train_x)
val_predict = NN.predict(val_x)
# training
plt.scatter([point[0] for point in train_y], [point[1] for point in train_y], c='b', alpha=0.05)
plt.scatter([point[0] for point in train_predict], [point[1] for point in train_predict], c='r', alpha=0.5)
plt.title('train')
plt.show()
# validation
plt.scatter([point[0] for point in val_y], [point[1] for point in val_y], c='y', alpha=0.5)
plt.scatter([point[0] for point in val_predict], [point[1] for point in val_predict], c='k', alpha=0.5)
plt.title('validation')
plt.show()
# test
# normalizziamo poi denormalizziamo il test
test_x = [np.array(row[:-2]) for row in test_data]
test_y = [np.array(row[-2:]) for row in test_data]

mean_test_arr = np.mean(test_x, axis=0)
std_test_arr = np.std(test_x, axis=0)
test_data = (test_x - mean_test_arr) / std_test_arr

predicted_test = NN.predict(test_x)

