from NN import *

'''
PROVA MONK 1
'''
data = np.genfromtxt("monk1.txt")
train_y = data[:, 0]
train_y = train_y.reshape((train_y.shape[0], 1))
train_x = data[:, 1:-1]
data_test = np.genfromtxt("test_monk1.txt")
test_y = data_test[:, 0]
test_y = test_y.reshape((test_y.shape[0], 1))
test_x = data_test[:, 1:-1]
# one of k
train_x = one_of_k(train_x)
test_x = one_of_k(test_x)
# grid seach
learning_rates = [0.05]
lambdas = [0]
alphas = [0.9]
neurons_per_layer = [10]
minibatch_sizes = [32]
layers_numbers = [1]
for neuron in neurons_per_layer:
    for layer in layers_numbers:
        for learning_rate in learning_rates:
            for Lambda in lambdas:
                for alpha in alphas:
                    for  minibatch_size in minibatch_sizes:
                        titolo = 'layer = ' + str(layer * [neuron]) + ', funzioni = ' + str(layer * ['tanh']) + ', learning_rate = ' + str(learning_rate) + ', Lambda = ' + str(Lambda) + ', alpha = ' + str(alpha) + ', minibatch_size = ' + str(minibatch_size)
                        print(titolo)
                        NN = NeuralNetwork(layer * [neuron], layer * ['tanh'], classification=True, learning_rate=learning_rate, Lambda=Lambda, alpha=alpha, toll=0.000001, n_init=1, max_epochs=150, minibatch_size=minibatch_size)
                        error_list, n_epochs, test_error_list, acc_list, test_acc_list = NN.fit(train_x, train_y, test_x, test_y)
                        print('train error = ' + str(error_list[-1]), 'test_error = ' + str(test_error_list[-1]), 'train accuracy = ' + str(acc_list[-1]), 'test accuracy = ' + str(test_acc_list[-1]))
                        plt.plot(range(n_epochs + 1), error_list)
                        plt.plot(range(n_epochs + 1), test_error_list, ls='dashed')
                        plt.legend(['train error', 'test error'])
                        plt.title('MSE')
                        plt.xlabel('number of epochs')
                        plt.ylabel('MSE')
                        plt.show()
                        #plt.savefig('C:/Users/danie/Desktop/Daniele/Laurea magistrale/Machine Learning/Machine-Learning-project/plot/MSE_' + titolo +'.png')
                        plt.close()
                        plt.plot(range(n_epochs + 1), acc_list)
                        plt.plot(range(n_epochs + 1), test_acc_list, ls='dashed')
                        plt.legend(['train accuracy', 'test accuracy'])
                        plt.title('accuracy')
                        plt.xlabel('number of epochs')
                        plt.ylabel('accuracy')
                        plt.show()
                        #plt.savefig('C:/Users/danie/Desktop/Daniele/Laurea magistrale/Machine Learning/Machine-Learning-project/plot/accuracy_' + titolo +'.png')
                        plt.close()

'''
PROVA MONK 2

data = np.genfromtxt("monk2.txt")
train_y = data[:, 0]
train_y = train_y.reshape((train_y.shape[0], 1))
train_x = data[:, 1:-1]
data_test = np.genfromtxt("test_monk2.txt")
test_y = data_test[:, 0]
test_y = test_y.reshape((test_y.shape[0], 1))
test_x = data_test[:, 1:-1]
# one of k
train_x = one_of_k(train_x)
test_x = one_of_k(test_x)
# grid seach
learning_rates = np.arange(0.01, 0.1, 0.01)
alphas = np.arange(0.1, 1, 0.1)
neurons_per_layer = [5, 10, 20, 30]
layers_numbers = [3, 4]
for neuron in neurons_per_layer:
    for layer in layers_numbers:
        for learning_rate in learning_rates:
            for alpha in alphas:
                titolo = 'layer = ' + str(layer * [neuron]) + ', funzioni = ' + str(layer * ['tanh']) + ', learning_rate = ' + str(learning_rate) + ', Lambda = 0' + ', alpha = ' + str(alpha) + ', minibatch_size = 8'
                print(titolo)
                NN = NeuralNetwork(layer * [neuron], layer * ['tanh'], classification=True, learning_rate=learning_rate, Lambda=0, alpha=alpha, toll=0.000001, n_init=1, max_epochs=100, minibatch_size=8)
                error_list, n_epochs, test_error_list, acc_list, test_acc_list = NN.fit(train_x, train_y, test_x, test_y)
                print('train error = ' + str(error_list[-1]), 'test_error = ' + str(test_error_list[-1]), 'train accuracy = ' + str(acc_list[-1]), 'test accuracy = ' + str(test_acc_list[-1]))
                plt.plot(range(n_epochs + 1), error_list)
                plt.plot(range(n_epochs + 1), test_error_list, ls='dashed')
                plt.legend(['train error', 'test error'])
                plt.title('MSE '+ titolo)
                plt.xlabel('number of epochs')
                plt.ylabel('MSE')
                plt.show()
                plt.plot(range(n_epochs + 1), acc_list)
                plt.plot(range(n_epochs + 1), test_acc_list, ls='dashed')
                plt.legend(['train accuracy', 'test accuracy'])
                plt.title('accuracy '+ titolo)
                plt.xlabel('number of epochs')
                plt.ylabel('accuracy')
                plt.show()
'''
'''
PROVA MONK 3

data = np.genfromtxt("monk3.txt")
train_y = data[:, 0]
train_y = train_y.reshape((train_y.shape[0], 1))
train_x = data[:, 1:-1]
data_test = np.genfromtxt("test_monk3.txt")
test_y = data_test[:, 0]
test_y = test_y.reshape((test_y.shape[0], 1))
test_x = data_test[:, 1:-1]
# one of k
train_x = one_of_k(train_x)
test_x = one_of_k(test_x)
# grid seach
learning_rates = [0.03]
lambdas = [0.2]
alphas = [0.7]
neurons_per_layer = [20]
minibatch_sizes = [128]
layers_numbers = [1]
for neuron in neurons_per_layer:
    for layer in layers_numbers:
        for learning_rate in learning_rates:
            for Lambda in lambdas:
                for alpha in alphas:
                    for  minibatch_size in minibatch_sizes:
                        titolo = 'layer = ' + str(layer * [neuron]) + ', funzioni = ' + str(layer * ['tanh']) + ', learning_rate = ' + str(learning_rate) + ', Lambda = ' + str(Lambda) + ', alpha = ' + str(alpha) + ', minibatch_size = ' + str(minibatch_size)
                        print(titolo)
                        NN = NeuralNetwork(layer * [neuron], layer * ['tanh'], classification=True, learning_rate=learning_rate, Lambda=Lambda, alpha=alpha, toll=0.000001, n_init=1, max_epochs=200, minibatch_size=minibatch_size)
                        error_list, n_epochs, test_error_list, acc_list, test_acc_list = NN.fit(train_x, train_y, test_x, test_y)
                        print('train error = ' + str(error_list[-1]), 'test_error = ' + str(test_error_list[-1]), 'train accuracy = ' + str(acc_list[-1]), 'test accuracy = ' + str(test_acc_list[-1]))
                        plt.plot(range(n_epochs + 1), error_list)
                        plt.plot(range(n_epochs + 1), test_error_list, ls='dashed')
                        plt.legend(['train error', 'test error'])
                        plt.title('MSE')
                        plt.xlabel('number of epochs')
                        plt.ylabel('MSE')
                        plt.show()
                        #plt.savefig('C:/Users/danie/Desktop/Daniele/Laurea magistrale/Machine Learning/Machine-Learning-project/plot/MSE_' + titolo +'.png')
                        plt.close()
                        plt.plot(range(n_epochs + 1), acc_list)
                        plt.plot(range(n_epochs + 1), test_acc_list, ls='dashed')
                        plt.legend(['train accuracy', 'test accuracy'])
                        plt.title('accuracy')
                        plt.xlabel('number of epochs')
                        plt.ylabel('accuracy')
                        plt.show()
                        #plt.savefig('C:/Users/danie/Desktop/Daniele/Laurea magistrale/Machine Learning/Machine-Learning-project/plot/accuracy_' + titolo +'.png')
                        plt.close()

'''