# Absztrakt ősosztályt definiálunk
# Ha egy osztály absztrakt, akkor nem példányosítjuk
# Különböző réteg típusokat, ez alapján hozzuk létre

class Layer:
    def __init__(self):
        self.input = None
        self.ouput = None
        
    def forward_propagation(self, input):
        raise NotImplementedError
    
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError