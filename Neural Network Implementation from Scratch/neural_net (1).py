from typing import Sequence
import numpy as np

class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.
    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        return np.dot(X, W) + b 
    
    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        return np.clip(X, 0, np.inf)

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.
        Parameters:
            X: the input data
        Returns:
            the output
        """
        X -= np.max(X)
        
        softmax_probability = np.exp(X)/np.sum(np.exp(X))
        
        return softmax_probability
                    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters:
            X: Input data of shape (N, D). 
        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of the network
        """
        
        self.outputs = {}

        linear_layer = self.linear(self.params['W1'], X, self.params['b1'])
        self.outputs['Input X for first layer'] = X

        layers_linear = [linear_layer]
        self.outputs['linear layer ' + str(1)] = linear_layer #first layer outputs
        
        layers_linear[-1] = self.relu(layers_linear[-1]) #RELU activation applied
        self.outputs['RELU layer ' + str(1)] = layers_linear[-1]

        for layer in range(2, self.num_layers):

            layers_linear.append(self.linear(self.params['W' + str(layer)], layers_linear[-1], self.params['b' + str(layer)]))
            self.outputs['linear layer ' + str(layer)] = layers_linear[-1]

            layers_linear[-1] = self.relu(layers_linear[-1]) #RELU activation applied
            self.outputs['RELU layer ' + str(layer)] = layers_linear[-1]
            
        #final layer transformation without RELU
        layers_linear.append(self.linear(self.params['W' + str(self.num_layers)], layers_linear[-1], self.params['b' + str(self.num_layers)])) 
        self.outputs['Final linear layer'] = layers_linear[-1]
                    
        #Applying Softmax 
        idx = list(range(len(layers_linear[-1])))
        
        softmax_output = np.array([self.softmax(layers_linear[-1][i]) for i in idx])
    
        self.outputs['Softmax Output'] = softmax_output  
        
        return softmax_output   #SoftMax applied to final layer having C classes for N samples resulting in an (N, C) matrix

    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Gradient of the Linear Layer.
        Parameters:
            X: the input data
            W: Weights
            b: Bias
        Returns:
            the local gradients of the linear layer wrt its parameters and inputs
        """
        grad_wrt_input = W.T   
                    
        grad_wrt_B = np.identity(b.shape[0])
                    
        return grad_wrt_input, grad_wrt_B

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Local Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            The Jacobian of local gradients of the ReLU layer
        """
        # TODO: implement me
        
        jacobian = np.eye(X.shape[0]) #computed for each row vector
        
        indicator = np.where(X>0, 1, 0)
                    
        np.fill_diagonal(jacobian, indicator) #final RELU gradient matrix
        
        return jacobian
    
    def CEloss_softmax_grad(self, X: np.ndarray,  y: int) -> np.ndarray:
        """ Gradient of CE loss with respect to input of softmax function computed directly 
            with an implicit Chain Rule applied as follows:
            
            de/dz = (de/dSoftmax) * (dSoftmax/dz)
        """         
        return np.array([X[i] - 1 if i == y else X[i] for i in range(len(X))])  #(1,C) vector gradient

    def backward(self, y: np.ndarray, reg: float = 0.0) -> float:
        """The backward pass for back-propagation to enable the compututation the gradients and losses.
        
        Parameters:
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            reg: Regularization strength
            
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        softmax_output = self.outputs['Softmax Output']
        
        #Total Cross-Entropy Loss with regularization
        initial_loss = 0
        m = y.shape[0]
        loss = -np.log(softmax_output[range(m), y] + 1e-50)
        initial_loss = np.sum(loss)/m
        
        for layer in range(1, self.num_layers + 1):
            
            initial_loss += (reg/2) * np.square(np.linalg.norm(self.params['W' + str(layer)]))
            self.gradients['W' + str(layer)] = 0
            self.gradients['b' + str(layer)] = 0
            
        #Initial Gradient        
            
        for samples in range(len(softmax_output)):
            
            initial_gradient = self.CEloss_softmax_grad(softmax_output[samples], y[samples])
            
            self.gradients['de/dz ' + str(samples)] = initial_gradient.reshape(1, -1)
            
            grad_wrt_input, grad_wrt_b = self.linear_grad(self.params['W'+str(self.num_layers)],self.outputs['RELU layer '+str(self.num_layers-1)][samples], self.params['b' + str(self.num_layers)])

            x = self.outputs['RELU layer ' + str(self.num_layers-1)][samples].reshape(1, -1)
            
            Wk_grad = np.dot(x.T, self.gradients['de/dz ' + str(samples)]) 
            self.gradients['W' + str(self.num_layers)] += (Wk_grad + (reg * self.params['W' + str(self.num_layers)]))/m
            
            bk_grad = np.sum(self.gradients['de/dz ' + str(samples)], axis = 0)
            self.gradients['b' + str(self.num_layers)] += bk_grad/m
            
            input_grad = grad_wrt_input
            chain_rule = [np.dot(initial_gradient, input_grad)]
            
            #Computing Gradients at each pass
            for layer in reversed(range(2, self.num_layers)):

                RELU_grad = self.relu_grad(self.outputs['linear layer ' + str(layer)][samples])
                
                chain_rule.append(np.dot(chain_rule[-1], RELU_grad))
                
                input_grads, bias_grad = self.linear_grad(self.params['W' + str(layer)], self.outputs['RELU layer ' + str(layer - 1)][samples], self.params['b' + str(layer)])
                x = self.outputs['RELU layer ' + str(layer-1)][samples].reshape(1, -1)
                
                W_grad = np.dot(x.T, chain_rule[-1].reshape(1,-1))
                self.gradients['W' + str(layer)] += (W_grad + (reg * self.params['W' + str(layer)]))/m
                
                b_grad = np.dot(chain_rule[-1], bias_grad)
                self.gradients['b' + str(layer)] += b_grad/m
                
                chain_rule.append(np.dot(chain_rule[-1], input_grads))
        
            RELU_grad = self.relu_grad(self.outputs['linear layer ' + str(1)][samples])
            
            final_chain_rule = np.dot(chain_rule[-1], RELU_grad)

            input_grads, bias_grad = self.linear_grad(self.params['W1'], self.outputs['Input X for first layer'][samples], self.params['b1'])
            
            x = self.outputs['Input X for first layer'][samples].reshape(1, -1)
        
            W_grad = np.dot(x.T, final_chain_rule.reshape(1, -1))
            self.gradients['W1'] += (W_grad + (reg * self.params['W1']))/m

            b_grad = np.dot(final_chain_rule, bias_grad)
            self.gradients['b1'] += b_grad/m
        
        return initial_loss

    def update(self, timestep: int, lr: float, opt: str, b1: float = 0.99, b2: float = 0.96, eps: float = 1e-8):
        
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            timestep: The training epoch under consideration
            lr: Learning rate
            opt: optimizer (setup for Adam and/or SGD)
            b1: beta 1 parameter
            b2: beta 2 parameter
            eps: epsilon to prevent division by zero
        """
        if opt == "SGD":
            
            for layer in range(1, self.num_layers + 1):
                self.params['W' + str(layer)] = self.params['W' + str(layer)] - (lr*self.gradients['W' + str(layer)])
                self.params['b' + str(layer)] = self.params['b' + str(layer)] - (lr*self.gradients['b' + str(layer)])
        else:
            
            for layer in range(1, self.num_layers + 1):
                
                steps_b = []
                
                m_b = np.zeros((self.params['b' + str(layer)].shape[0], ))
                v_b = np.zeros((self.params['b' + str(layer)].shape[0], ))
                
                steps_b = [[m_b, v_b]] + steps_b
                
                m_b_t = b1*steps_b[-1][0] + (1-b1) * self.gradients['b' + str(layer)]
                
                v_b_t = b2*steps_b[-1][1] + (1-b2) * (self.gradients['b' + str(layer)] * self.gradients['b' + str(layer)])
                
                m_t_hat = m_b_t/(1 - (b1**timestep))
                v_t_hat = v_b_t/(1 - (b2**timestep))
                
                b_grad_prev = self.params['b' + str(layer)]
                
                self.params['b' + str(layer)] -= (lr * m_t_hat/(np.sqrt(v_t_hat)+eps))
                    
                steps_b.append([m_b_t, v_b_t])
                
                for idx in range(self.params['W' + str(layer)].shape[0]):
                    
                    steps_W = []
                
                    m_W = np.zeros((self.params['W' + str(layer)].shape[1], ))
                    v_W = np.zeros((self.params['W' + str(layer)].shape[1], ))

                    steps_W = [[m_W, v_W]] + steps_W

                    m_W_t = b1*steps_W[-1][0] + (1-b1) * self.gradients['W' + str(layer)][idx, :]
                    
                    v_W_t = b2*steps_W[-1][1] + (1-b2) * (self.gradients['W' + str(layer)][idx, :]*self.gradients['W' + str(layer)][idx, :])
                    
                    m_t_hat = m_W_t/(1 - (b1**timestep))
                    v_t_hat = v_W_t/(1 - (b2**timestep))

                    W_grad_prev = self.params['W' + str(layer)][idx, :]
                    
                    self.params['W' + str(layer)][idx, :] -= (lr * m_t_hat/(np.sqrt(v_t_hat)+eps))
                        
                    steps_W.append([m_W_t, v_W_t])
                    
        pass