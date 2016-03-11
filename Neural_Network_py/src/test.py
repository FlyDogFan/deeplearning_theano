import mnist_loader
import neural_network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = neural_network.Network([784, 150, 10])
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, evaluation_data=validation_data, 
        monitor_evaluation_accuracy=True, monitor_training_cost=True)
