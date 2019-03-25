from NN import *

def one_of_k(data):
    dist_values = np.array([np.unique(data[:, i]) for i in range(data.shape[1])])
    new_data = []
    First_rec = True
    for record in data:
        new_record = []
        First = True
        indice = 0
        for attribute in record:
            new_attribute = np.zeros(len(dist_values[indice]), dtype=int)
            for j in range(len(dist_values[indice])):
                if dist_values[indice][j] == attribute:
                    new_attribute[j] += 1
            if First:
                new_record = new_attribute
                First = False
            else:
                new_record = np.concatenate((new_record, new_attribute), axis=0)
            indice += 1
        if First_rec:
            new_data = np.array([new_record])
            First_rec = False
        else:
            new_data = np.concatenate((new_data, np.array([new_record])), axis=0)
    return new_data

'''
PROVA MONK 1
'''
data = np.genfromtxt("monk1.txt")
# splitting in test and train, after we shuffle the dataset
shuffle(data)
train_percentage = 0.8
n_train = round(data.shape[0] * train_percentage)
train = data[:n_train, :]
val = data[n_train:, :]
# splitting in attributes and class
train_y = train[:, 0]
train_y = train_y.reshape((train_y.shape[0], 1))
train_x = train[:, 1:-1]
val_y = val[:, 0]
val_y = val_y.reshape((val_y.shape[0], 1))
val_x = val[:, 1:-1]
# test_data
data_test = np.genfromtxt("test_monk1.txt")
test_y = data_test[:, 0]
test_y = test_y.reshape((test_y.shape[0], 1))
test_x = data_test[:, 1:-1]
# one of k
train_x = one_of_k(train_x)
val_x = one_of_k(val_x)
test_x = one_of_k(test_x)
# grid seach
learning_rates = [0.1] #[0.01, 0.03, 0.07, 0.1]
lambdas = [0] 
alphas = [0.8] #[0.7, 0.8, 0.9]
neurons_per_layer = [15] #[5, 10, 15, 17, 20]
minibatch_sizes = [32] #[32, 64, None]
layers_numbers = [1] #[1, 2, 3]
algorithm = 'SGD'
for neuron in neurons_per_layer:
    for layer in layers_numbers:
        for learning_rate in learning_rates:
            for Lambda in lambdas:
                for alpha in alphas:
                    for  minibatch_size in minibatch_sizes:
                        titolo = 'layer = ' + str(layer * [neuron]) + ', funzioni = ' + str(layer * ['tanh']) + ', learning_rate = ' + str(learning_rate) + ', Lambda = ' + str(Lambda) + ', alpha = ' + str(alpha) + ', minibatch_size = ' + str(minibatch_size) + ', algorithm = ' + algorithm
                        print(titolo)
                        NN = NeuralNetwork(layer * [neuron], layer * ['tanh'], classification=True, learning_rate=learning_rate, Lambda=Lambda, alpha=alpha, toll=0.000001, n_init=10, max_epochs=100, minibatch_size=minibatch_size, algorithm=algorithm)
                        NN.fit(train_x, train_y)
                        print('MSE training set = ' + str(NN.score(train_x, train_y)), 'MSE validation set = ' + str(NN.score(val_x, val_y)))
                        pred_class_train = [round(elem[0]) for elem in NN.predict(train_x)]
                        pred_class_val = [round(elem[0]) for elem in NN.predict(val_x)]
                        train_acc = 1 - sum([1 if pred_class_train[i] != train_y[i][0] else 0 for i in range(len(pred_class_train))]) / len(pred_class_train)
                        val_acc = 1 - sum([1 if pred_class_val[i] != val_y[i][0] else 0 for i in range(len(pred_class_val))]) / len(pred_class_val)
                        print('Accuracy training set = ' + str(train_acc), 'Accuracy validation set = ' + str(val_acc))
# cross validation, commentata
'''
data = np.genfromtxt("monk1.txt")
train_and_val_data_x = data[:, 1:-1]
train_and_val_data_x = one_of_k(train_and_val_data_x)
train_and_val_data_y = train_and_val_data[:, 0]
train_and_val_data_y = train_and_val_data_y.reshape((train_and_val_data_y.shape[0], 1))
cv = NN.k_fold_cv(train_and_val_data_x, train_and_val_data_y)
print(np.mean(cv), np.std(cv))
# OUTPUT: 0.0026   0.0034
'''                        
print('MSE test set', NN.score(test_x, test_y))
pred_class_test = [round(elem[0]) for elem in NN.predict(test_x)]
test_acc = 1 - sum([1 if pred_class_test[i] != test_y[i][0] else 0 for i in range(len(pred_class_test))]) / len(pred_class_test)
print('Accuracy test set', test_acc)

'''
PROVA MONK 2
'''
data = np.genfromtxt("monk2.txt")
# splitting in test and train, after we shuffle the dataset
shuffle(data)
train_percentage = 0.8
n_train = round(data.shape[0] * train_percentage)
train = data[:n_train, :]
val = data[n_train:, :]
# splitting in attributes and class
train_y = train[:, 0]
train_y = train_y.reshape((train_y.shape[0], 1))
train_x = train[:, 1:-1]
val_y = val[:, 0]
val_y = val_y.reshape((val_y.shape[0], 1))
val_x = val[:, 1:-1]
# test_data
data_test = np.genfromtxt("test_monk2.txt")
test_y = data_test[:, 0]
test_y = test_y.reshape((test_y.shape[0], 1))
test_x = data_test[:, 1:-1]
# one of k
train_x = one_of_k(train_x)
val_x = one_of_k(val_x)
test_x = one_of_k(test_x)
# grid seach
learning_rates = [0.1] #[0.01, 0.03, 0.07, 0.1]
alphas = [0.8] #[0.7, 0.8, 0.9]
lambdas = [0]
neurons_per_layer = [10] #[5, 10, 15, 17, 20]
minibatch_sizes = [32] #[32, 64, None]
layers_numbers = [1] #[1, 2, 3]
algorithm = 'SGD'
for neuron in neurons_per_layer:
    for layer in layers_numbers:
        for learning_rate in learning_rates:
            for Lambda in lambdas:
                for alpha in alphas:
                    for  minibatch_size in minibatch_sizes:
                        titolo = 'layer = ' + str(layer * [neuron]) + ', funzioni = ' + str(layer * ['tanh']) + ', learning_rate = ' + str(learning_rate) + ', Lambda = ' + str(Lambda) + ', alpha = ' + str(alpha) + ', minibatch_size = ' + str(minibatch_size) + ', algorithm = ' + algorithm
                        print(titolo)
                        NN = NeuralNetwork(layer * [neuron], layer * ['tanh'], classification=True, learning_rate=learning_rate, Lambda=Lambda, alpha=alpha, toll=0.000001, n_init=10, max_epochs=100, minibatch_size=minibatch_size, algorithm=algorithm)
                        NN.fit(train_x, train_y)
                        print('MSE training set = ' + str(NN.score(train_x, train_y)), 'MSE validation set = ' + str(NN.score(val_x, val_y)))
                        pred_class_train = [round(elem[0]) for elem in NN.predict(train_x)]
                        pred_class_val = [round(elem[0]) for elem in NN.predict(val_x)]
                        train_acc = 1 - sum([1 if pred_class_train[i] != train_y[i][0] else 0 for i in range(len(pred_class_train))]) / len(pred_class_train)
                        val_acc = 1 - sum([1 if pred_class_val[i] != val_y[i][0] else 0 for i in range(len(pred_class_val))]) / len(pred_class_val)
                        print('Accuracy training set = ' + str(train_acc), 'Accuracy validation set = ' + str(val_acc))
# cross validation, commentata
'''
data = np.genfromtxt("monk2.txt")
train_and_val_data_x = data[:, 1:-1]
train_and_val_data_x = one_of_k(train_and_val_data_x)
train_and_val_data_y = train_and_val_data[:, 0]
train_and_val_data_y = train_and_val_data_y.reshape((train_and_val_data_y.shape[0], 1))
cv = NN.k_fold_cv(train_and_val_data_x, train_and_val_data_y)
print(np.mean(cv), np.std(cv))
# OUTPUT: 0.0003   7.5e-05
'''            
print('MSE test set', NN.score(test_x, test_y))
pred_class_test = [round(elem[0]) for elem in NN.predict(test_x)]
test_acc = 1 - sum([1 if pred_class_test[i] != test_y[i][0] else 0 for i in range(len(pred_class_test))]) / len(pred_class_test)
print('Accuracy test set', test_acc)

'''
PROVA MONK 3
'''
data = np.genfromtxt("monk3.txt")
# splitting in test and train, after we shuffle the dataset
shuffle(data)
train_percentage = 0.8
n_train = round(data.shape[0] * train_percentage)
train = data[:n_train, :]
val = data[n_train:, :]
# splitting in attributes and class
train_y = train[:, 0]
train_y = train_y.reshape((train_y.shape[0], 1))
train_x = train[:, 1:-1]
val_y = val[:, 0]
val_y = val_y.reshape((val_y.shape[0], 1))
val_x = val[:, 1:-1]
# test_data
data_test = np.genfromtxt("test_monk3.txt")
test_y = data_test[:, 0]
test_y = test_y.reshape((test_y.shape[0], 1))
test_x = data_test[:, 1:-1]
# one of k
train_x = one_of_k(train_x)
val_x = one_of_k(val_x)
test_x = one_of_k(test_x)
# grid seach
learning_rates = [0.01] #[0.01, 0.03, 0.07, 0.1]
lambdas = [0.2] #[0.01, 0.1, 0.2]
alphas = [0.8] #[0.7, 0.8, 0.9]
neurons_per_layer = [10] #[5, 10, 15, 17, 20]
minibatch_sizes = [64] #[32, 64, None]
layers_numbers = [3] #[1, 2, 3]
algorithm = 'SGD'
for neuron in neurons_per_layer:
    for layer in layers_numbers:
        for learning_rate in learning_rates:
            for Lambda in lambdas:
                for alpha in alphas:
                    for  minibatch_size in minibatch_sizes:
                        titolo = 'layer = ' + str(layer * [neuron]) + ', funzioni = ' + str(layer * ['tanh']) + ', learning_rate = ' + str(learning_rate) + ', Lambda = ' + str(Lambda) + ', alpha = ' + str(alpha) + ', minibatch_size = ' + str(minibatch_size) + ', algorithm = ' + algorithm
                        print(titolo)
                        NN = NeuralNetwork(layer * [neuron], layer * ['tanh'], classification=True, learning_rate=learning_rate, Lambda=Lambda, alpha=alpha, toll=0.000001, n_init=10, max_epochs=100, minibatch_size=minibatch_size, algorithm=algorithm)
                        NN.fit(train_x, train_y)
                        print('MSE training set = ' + str(NN.score(train_x, train_y)), 'MSE validation set = ' + str(NN.score(val_x, val_y)))
                        pred_class_train = [round(elem[0]) for elem in NN.predict(train_x)]
                        pred_class_val = [round(elem[0]) for elem in NN.predict(val_x)]
                        train_acc = 1 - sum([1 if pred_class_train[i] != train_y[i][0] else 0 for i in range(len(pred_class_train))]) / len(pred_class_train)
                        val_acc = 1 - sum([1 if pred_class_val[i] != val_y[i][0] else 0 for i in range(len(pred_class_val))]) / len(pred_class_val)
                        print('Accuracy training set = ' + str(train_acc), 'Accuracy validation set = ' + str(val_acc))
# cross validation, commentata
'''
data = np.genfromtxt("monk3.txt")
train_and_val_data_x = data[:, 1:-1]
train_and_val_data_x = one_of_k(train_and_val_data_x)
train_and_val_data_y = train_and_val_data[:, 0]
train_and_val_data_y = train_and_val_data_y.reshape((train_and_val_data_y.shape[0], 1))
cv = NN.k_fold_cv(train_and_val_data_x, train_and_val_data_y)
print(np.mean(cv), np.std(cv))
# OUTPUT: 0.0386   0.0022
'''
# test set  
print('MSE test set', NN.score(test_x, test_y))
pred_class_test = [round(elem[0]) for elem in NN.predict(test_x)]
test_acc = 1 - sum([1 if pred_class_test[i] != test_y[i][0] else 0 for i in range(len(pred_class_test))]) / len(pred_class_test)
print('Accuracy test set', test_acc)