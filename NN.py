# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:20:03 2018

@authors: senes, fibs1993
"""

import numpy as np
from numpy.random import shuffle, randn, uniform
import matplotlib.pyplot as plt
from math import e
from math import exp
from math import sqrt
from math import tanh
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

def my_prod(matrix_list1, matrix_list2):
    return [matrix_list1[i] * matrix_list2[i] for i in range(len(matrix_list1))]

def my_div(matrix_list1, matrix_list2):
    return [matrix_list1[i] / matrix_list2[i] for i in range(len(matrix_list1))]

def my_sqrt(matrix_list1):
    return [np.sqrt(matrix_list1[i]) for i in range(len(matrix_list1))]

def my_prod_per_scal(scalar, matrix_list):
    return [scalar * matrix for matrix in matrix_list]

def my_sum_per_scal(scalar, matrix_list):
    return [scalar + matrix for matrix in matrix_list]

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
    def __init__(self, hidden_layers, act_functs, toll=0.1, learning_rate=0.0005, type_lr='constant' , alpha = 0, minibatch_size=None, max_epochs=200, Lambda=0.001,  n_init=5, classification=False, algorithm='ADAM'):
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
        self.learning_rate = None
        self.initial_lrate = learning_rate
        self.type_lr = type_lr
        self.alpha = alpha
        self.minibatch_size = minibatch_size
        self.max_epochs = max_epochs
        self.Lambda = Lambda
        self.n_init = n_init
        self.algorithm = algorithm

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
                    self.weights[i][j][k] = randn() * np.sqrt(1 / len(self.weights[i][j]))
                    
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
        layers_list = [train_x.shape[1]] + list(self.hidden_layers) + [train_y.shape[1]]
        self.weights = [np.empty((layers_list[i], layers_list[i-1] + 1), dtype='float32') for i in range(1, len(layers_list))]
        self.deltas = [np.empty(n_neuron, dtype='float32') for n_neuron in layers_list]

        # ora inizia l'algoritmo
        if self.algorithm == 'ADAM':
            min_error = float('inf')
            if self.minibatch_size == None:
                self.minibatch_size = train_x.shape[0]
            # i pesi vanno inizializzati più volte
            # ogni volta che li inizializziamo facciamo ripartire l'algoritmo vero e proprio
            # memorizziamo l'errore minimo di ogni tentativo e i pesi migliori
            for _ in range(self.n_init):
                #print('inizializzazione', n + 1)
                curr_error = float('inf')
                gradient = [np.zeros_like(layer) for layer in self.weights]
                self._init_weights()

                t = 0
                m = [np.zeros_like(layer) for layer in self.weights]
                v = [np.zeros_like(layer) for layer in self.weights]
                m_ = [np.zeros_like(layer) for layer in self.weights]
                v_ = [np.zeros_like(layer) for layer in self.weights] 
                eps = 10**(-8)
                beta1 = 0.9
                beta2 = 0.999

                for n_epochs in range(self.max_epochs):
                    # inizializzo / decremento il learning rate
                    # constant
                    if self.type_lr == 'constant':
                        self.learning_rate = self.initial_lrate
                    # step decay
                    if type(self.type_lr) == tuple:
                        decay_factor = self.type_lr[0]
                        step_size = self.type_lr[1]
                        self.learning_rate = self.initial_lrate * (decay_factor ** np.floor(n_epochs / step_size))
                    # exponential decay
                    if type(self.type_lr) == float:
                        self.learning_rate = self.initial_lrate * exp(-(self.type_lr * n_epochs))

                    # calcolo del gradiente, sommando tutti i risultati di ogni backprop
                    for index, pattern in enumerate(train_x):
                        outputNN = self._forward(pattern)
                        gradient = my_sum(gradient, self._backward(outputNN, train_y[index]))
                        if index != 0 and index % self.minibatch_size == 0:
                            if self.Lambda != 0:
                                gradient = my_sum(gradient, my_prod_per_scal(-self.Lambda, self.weights))
                            t = t + 1
                            m = my_sum(my_prod_per_scal(beta1, m), my_prod_per_scal((1 - beta1), gradient))
                            v = my_sum(my_prod_per_scal(beta2, v), my_prod_per_scal((1 - beta2), my_prod(gradient, gradient)))
                            m_ = my_prod_per_scal(1 / (1 - beta1**t), m)
                            v_ = my_prod_per_scal(1 / (1 - beta2**t), v)
                            num = my_prod_per_scal(self.learning_rate, m_)
                            div = my_sum_per_scal(eps, my_sqrt(v_))
                            self.weights = my_sum(self.weights, my_div(num, div))

                            gradient = my_prod_per_scal(self.alpha, gradient)
                    
                    # dopo aver visto tutti i pattern bisogna aggiornare i pesi
                    if self.Lambda != 0:
                        gradient = my_sum(gradient, my_prod_per_scal(-self.Lambda, self.weights))
                    t = t + 1
                    m = my_sum(my_prod_per_scal(beta1, m), my_prod_per_scal((1 - beta1), gradient))
                    v = my_sum(my_prod_per_scal(beta2, v), my_prod_per_scal((1 - beta2), my_prod(gradient, gradient)))
                    m_ = my_prod_per_scal(1 / (1 - beta1**t), m)
                    v_ = my_prod_per_scal(1 / (1 - beta2**t), v)
                    num = my_prod_per_scal(self.learning_rate, m_)
                    div = my_sum_per_scal(eps, my_sqrt(v_))
                    self.weights = my_sum(self.weights, my_div(num, div))

                    gradient = my_prod_per_scal(0, gradient)

                    # calcolo errore
                    curr_error = self.score(train_x, train_y)
                    #print(curr_error)

                    if curr_error < self.toll:
                        break

                # alla fine dell'allenamento, se abbiamo ottenuto risultati migliori aggiorniamo min_error e best_weights
                if curr_error <= min_error:
                    min_error = curr_error
                    best_weights = deepcopy(self.weights)

            self.weights = best_weights
            return 
        
        if self.algorithm == 'SGD':
            # se minibatch_size = None ---> versione batch dell'algoritmo, quindi minibatch_size = len(train_x)
            if self.minibatch_size == None:
                self.minibatch_size = train_x.shape[0]

            # ora inizia l'algoritmo
            min_error = float('inf')
            # i pesi vanno inizializzati più volte
            # ogni volta che li inizializziamo facciamo ripartire l'algoritmo vero e proprio
            # memorizziamo l'errore minimo di ogni tentativo e i pesi migliori
            for _ in range(self.n_init):
                #print('inizializzazione', n + 1)
                curr_error = float('inf')
                gradient = [np.zeros_like(layer) for layer in self.weights]
                self._init_weights()

                for n_epochs in range(self.max_epochs):
                    # inizializzo / decremento il learning rate
                    # constant
                    if self.type_lr == 'constant':
                        self.learning_rate = self.initial_lrate
                    # step decay
                    if type(self.type_lr) == tuple:
                        decay_factor = self.type_lr[0]
                        step_size = self.type_lr[1]
                        self.learning_rate = self.initial_lrate * (decay_factor ** np.floor(n_epochs / step_size))
                    # exponential decay
                    if type(self.type_lr) == float:
                        self.learning_rate = self.initial_lrate * exp(-(self.type_lr * n_epochs))

                    # calcolo del gradiente, sommando tutti i risultati di ogni backprop
                    for index, pattern in enumerate(train_x):
                        outputNN = self._forward(pattern)
                        gradient = my_sum(gradient, self._backward(outputNN, train_y[index]))
                        # dopo minibatch_size passi aggiorniamo i pesi e reinizializziamo il gradiente
                        # NB: la regolarizzazione viene fatta in update_weights
                        if index != 0 and index % self.minibatch_size == 0:
                            self._update_weights(gradient)
                            # reset the gradient, to 0 if no momentum(alpha = 0)
                            # to alpha times the old gradient otherwise
                            gradient = my_prod_per_scal(self.alpha, gradient)

                    # dopo aver visto tutti i pattern bisogna nuovamente aggiornare i pesi
                    self._update_weights(gradient)
                    gradient = my_prod_per_scal(self.alpha, gradient)
                    # calcolo errore
                    curr_error = self.score(train_x, train_y)
                    #print(curr_error)

                    if curr_error < self.toll:
                        break

                # alla fine dell'allenamento, se abbiamo ottenuto risultati migliori aggiorniamo min_error e best_weights
                if curr_error <= min_error:
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
        len_data = data.shape[0]
        for i in range(len_data):
            outputNN = self._forward(data[i])
            output_arr.append(outputNN)
        return np.array(output_arr)

    def score(self, data_x, data_y):
    # ritorna l'accuracy oppure il mean squared error
        # errore se la rete non è stata fittata --> non si conosce il numero di input e output
        if not self.weights:
            raise UntrainedError('La rete deve prima essere allenata!')

        predicted_y = self.predict(data_x)
        # NB predicted_y è un array, così come data_y
        if self.act_functs[-1] == sigmoid:
            # MSE per classificazione
            error_list = [sum((prediction - data_y[index])**2) for index, prediction in enumerate(predicted_y)]
        else:
            # euclidean error per regressione
            error_list = [np.sqrt(sum((prediction - data_y[index])**2)) for index, prediction in enumerate(predicted_y)]

        return sum(error_list) / len(error_list)
    
    def k_fold_cv(self, data_x, data_y, k=5):
    # ritorna una lista di score, uno per ogni tentativo

        # calcoliamo la lunghezza di ogni divisione del dataset
        divided_data_size = data_x.shape[0] // k
        # np.split divide il dataset a seconda degli indici che gli passiamo nella lista (secondo parametro)
        # quindi list_subdata è una lista di un np.ndarray bidimensionali (attributi in colonna, record in riga)
        list_subdata_x = np.split(data_x, [i * divided_data_size for i in range(1, k)])
        list_subdata_y = np.split(data_y, [i * divided_data_size for i in range(1, k)])
        error_list = []
        for i in range(k):
            # splitting in test and train
            # il test set è semplicemente l'i-esimo elemento della lista delle porzioni del dataset
            val_x = list_subdata_x[i]
            val_y = list_subdata_y[i] 
            # il training set è l'array dei record presenti in tutti gli altri elementi della lista delle porzioni
            train_x = np.array([])#.reshape((0, val_x.shape[1]))
            train_y = np.array([])
            for j in range(k):
                if j != i:
                    for rows in list_subdata_x[j]:
                        train_x = np.vstack([train_x, rows]) if train_x.size else rows
                    for rows in list_subdata_y[j]:
                        train_y = np.vstack([train_y, rows]) if train_y.size else rows
            # fit the neural network
            self.fit(train_x, train_y)
            # calculate test error and append it to the result
            test_error = self.score(val_x, val_y)
            error_list.append(test_error)
        
        return error_list

def funzione(x):
    if x < -17.825393:
        return -1.1334002237288088 * (x + 28.835118) - 2.531668 
    return 0.8568463444857496 * (x + 9.091727) - 7.759054


if __name__ == '__main__':
    # eliminiamo la colonna dell'indice
    data = np.genfromtxt("ML-CUP18-TR.csv", delimiter=',')[:, 1:]
    # splitting in test and train, after we shuffle the dataset
    shuffle(data)
    train_and_val_percentage = 0.75
    n_train_and_val = round(len(data) * train_and_val_percentage)
    train_and_val_data = data[:n_train_and_val, :]
    test_data = data[n_train_and_val:, :]
    # splitting in train and validation
    train_percentage = 0.75
    n_train = round(len(train_and_val_data) * train_percentage)
    train_data = train_and_val_data[:n_train, :]
    val_data = train_and_val_data[n_train:, :]
    # splitting in train attributes, train target, test attr and test target
    train_x = train_data[:, :-2]
    train_y = train_data[:, -2:]
    val_x = val_data[:, :-2]
    val_y = val_data[:, -2:]
    # grid seach fatta su un file jupyter, per il codice vedere monk.py
    learning_rate = 0.00005
    Lambda = 0.1
    alpha = 0.8
    neuron = 50
    minibatch_size = 128
    layer = 2
    type_lr = 'constant'
    algorithm = 'SGD'
    titolo = 'layer = ' + str(layer * [neuron]) + ', funzioni = ' + str((layer) * ['tanh']) + ', learning_rate = ' + str(learning_rate) + ', Lambda = ' + str(Lambda) + ', alpha = ' + str(alpha) + ', minibatch_size = ' + str(minibatch_size) + ', algorithm = ' + algorithm
    print(titolo)
    NN = NeuralNetwork((layer) * [neuron], (layer) * ['tanh'], learning_rate=learning_rate, type_lr=type_lr, Lambda=Lambda, alpha=alpha, toll=0.000001, n_init=1, max_epochs=300, minibatch_size=minibatch_size, algorithm=algorithm)
    NN.fit(train_x, train_y)
    print('train error = ' + str(NN.score(train_x, train_y)), 'validation_error = ' + str(NN.score(val_x, val_y)))
    # cross validation, commentata perché prende troppo tempo
    '''
    data = np.genfromtxt("ML-CUP18-TR.csv", delimiter=',')[:, 1:]
    train_and_val_data_x = train_and_val_data[:, :-2]
    train_and_val_data_y = train_and_val_data[:, -2:]
    cv = NN.k_fold_cv(train_and_val_data_x, train_and_val_data_y, k=10)
    print(np.mean(cv), np.std(cv))
    OUTPUT: 1.1787   0.1017
    '''
    # test
    # rialleniamo la rete su tutto il training e validation set, che si chiamava train_and_val_data
    train_x = train_and_val_data[:, :-2]
    train_y = train_and_val_data[:, -2:]
    NN.fit(train_x, train_y)
    # ricorda test_data è la porzione di dataset non toccata
    test_x = test_data[:, :-2]
    test_y = test_data[:, -2:]
    # compute error
    test_error = NN.score(test_x, test_y)
    print('test_error', test_error)
    # predetti
    test_predict = NN.predict(test_x)
    #plot result
    plt.scatter([point[0] for point in test_y], [point[1] for point in test_y], c='y', alpha=0.5)
    plt.scatter([point[0] for point in test_predict], [point[1] for point in test_predict], c='k', alpha=0.5)
    plt.title('test')
    plt.show()
    
    '''
    # eliminiamo la colonna dell'indice
    data = np.genfromtxt("ML-CUP18-TR.csv", delimiter=',')[:, 1:]
    # splitting in test and train, after we shuffle the dataset
    shuffle(data)
    train_and_val_percentage = 0.7
    n_train_and_val = round(len(data) * train_and_val_percentage)
    train_and_val_data = data[:n_train_and_val, :]
    test_data = data[n_train_and_val:, :]
    # splitting in train and validation
    train_percentage = 0.7
    n_train = round(len(train_and_val_data) * train_percentage)
    train_data = train_and_val_data[:n_train, :]
    val_data = train_and_val_data[n_train:, :]
    # splitting in train attributes, train target, test attr and test target
    train_x = train_data[:, :-2]
    train_y = train_data[:, -1].reshape((train_data.shape[0], 1))
    val_x = val_data[:, :-2]
    val_y = val_data[:, -1].reshape((val_data.shape[0], 1))
    Y_tr = train_data[:, -2:]
    Y = val_data[:, -2:]
    # data normalization
    # Z-score normalization
    train_x = (train_x - np.mean(train_x, axis=0)) / np.std(train_x, axis=0)
    val_x = (val_x - np.mean(val_x, axis=0)) / np.std(val_x, axis=0)
    # normalization in [-1, 1]
    #for i in range(train_x.shape[1]):
    #    train_x[:, i] = 2 * ((train_x[:, i] - train_x[:, i].min()) / (train_x[:, i].max() - train_x[:, i].min())) - 1
    #    val_x[:, i] = 2 * ((val_x[:, i] - val_x[:, i].min()) / (val_x[:, i].max() - val_x[:, i].min())) - 1
    # grid seach
    learning_rates = [0.005]
    lambdas = [0.0]
    alphas = [0]
    neurons_per_layer = [20]
    minibatch_sizes = [32]
    layers_numbers = [1]
    type_lr = 'constant'
    for neuron in neurons_per_layer:
        for layer in layers_numbers:
            for learning_rate in learning_rates:
                for Lambda in lambdas:
                    for alpha in alphas:
                        for  minibatch_size in minibatch_sizes:
                            titolo = 'layer = ' + str(layer * [neuron]) + ', funzioni = ' + str((layer) * ['tanh']) + ', learning_rate = ' + str(learning_rate) + ', Lambda = ' + str(Lambda) + ', alpha = ' + str(alpha) + ', minibatch_size = ' + str(minibatch_size) + ', algorithm = ADAM'
                            print(titolo)
                            NN = NeuralNetwork((layer) * [neuron], (layer) * ['tanh'], learning_rate=learning_rate, type_lr=type_lr, Lambda=Lambda, alpha=alpha, toll=0.000001, n_init=1, max_epochs=100, minibatch_size=minibatch_size, algorithm='ADAM')
                            error_list, n_epochs, test_error_list, acc_list, test_acc_list = NN.fit(train_x, train_y, val_x, val_y)
                            print('train error = ' + str(error_list[-1]), 'test_error = ' + str(test_error_list[-1]), 'min test err = ' + str(min(test_error_list)))
                            print('errore previsto train = ' + str(error_list[-1] + funzione(error_list[-1])), '    errore previsto test = ' + str(test_error_list[-1] + funzione(test_error_list[-1])))
                            plt.plot(range(n_epochs + 1), error_list)
                            plt.plot(range(n_epochs + 1), test_error_list, ls='dashed')
                            plt.ylim((-0.5, 10))
                            plt.legend(['train error', 'test error'])
                            plt.title('MSE')
                            plt.xlabel('number of epochs')
                            plt.ylabel('MSE')
                            plt.show()
                            #plt.savefig('C:/Users/danie/Desktop/Daniele/Laurea magistrale/Machine Learning/Machine-Learning-project/plot/MSE_' + titolo +'.png')
                            plt.close()
                            pred_tr = np.array([np.array([funzione(y), y]) for y in NN.predict(train_x)])
                            pred_tr = pred_tr.reshape((pred_tr.shape[0], pred_tr.shape[1]))
                            pred = np.array([np.array([funzione(y), y]) for y in NN.predict(val_x)])
                            pred = pred.reshape((pred.shape[0], pred.shape[1]))
                            print('ERRORE TRAIN = ', sum(sum((Y_tr - pred_tr)**2)) / Y_tr.shape[0], '   ERRORE TEST = ', sum(sum((Y - pred)**2)) / Y.shape[0])
'''