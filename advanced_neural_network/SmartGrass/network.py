class Network():
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_deriv = None
        
    def add(self, layer):
        self.layers.append(layer)
        
    def use_loss(self, loss, loss_deriv):
        self.loss = loss
        self.loss_deriv = loss_deriv
        
    def predict(self, input):
        result = []
        for i in range(len(input)):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result
    
    def fit(self, train_x, train_y, epochs, learning_rate):
        for i in range(epochs):
            error_in_epoch = 0
            for j in range(len(train_x)):
                output = train_x[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                error_in_epoch += self.loss(output, train_y[j])
                error = self.loss_deriv(output, train_y[j])
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            error_in_epoch /= len(train_x)
            print(f"Epoch no. {i+1}, error: {error_in_epoch}")
                
            