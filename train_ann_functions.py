# Load modules
import numpy as np
from pandas import DataFrame

from keras.layers import Input, Dense, Concatenate, Lambda, Softmax
from keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam
from keras import Model
import keras
from typing import Any
import tensorflow as tf

# Class of ASS-NN
class AsuSharedNN:
    # Init function
    def __init__(
        self,
        topology: tuple = (5,),
        activation: str = 'relu',
        optimiser: str = 'adam',
        regularisation: str = None,
        hidden_bias: bool = False,
        output_bias: bool = False,
        learning_rate: float = 1e-3,
        from_logits: bool = False
        ):
        
        # Initialise class
        self.is_trained = False
        self.topology = topology
        self.activation = activation
        self.regularisation = regularisation
        self.hidden_bias = hidden_bias
        self.output_bias = output_bias
        self.from_logits = from_logits

        if optimiser == 'adam':
            self.optimiser = Adam(learning_rate=learning_rate)
        elif optimiser == 'sgd':
            self.optimiser = SGD(learning_rate=learning_rate)

    # Call function
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        assert self.is_trained, "Model is not trained (yet)"
        return self.model(*args,**kwds)

    # Fit function
    def fit(
            self,
            X,
            y,
            X_loc,
            X_shared,
            Z_loc = None,
            epochs: int = 10000,
            batch_size: float = 100,
            validation_sample: tuple = None,
            early_stopping: bool = False,
            validation_split: float = None,
            min_delta: float = 1e-6,
            patience: float = 6,
            verbose: bool = False
            ):
        
        # Set scalars
        assert len(np.unique(y)) > 1, 'Number of alternatives must be higher than 1'
        self.J = len(np.unique(y))

        assert len(X_loc) == X.shape[1], 'X_loc must be of length equal to the number of columns of X'
        assert len(X_shared) == X.shape[1], 'X_shared must be of length equal to the number of columns of X'

        # Set input layer
        input_layer = Input(shape=X.shape[1], name = 'input')

        # Set shared separators and mirror layer
        assert np.sum(X_shared) > 0, 'Number of shared inputs is zero. Why not use another ANN?'
        assert np.sum(X_shared) % self.J == 0, 'Number of shared inputs must be the same per alternative'
        K_shared = int(np.sum(X_shared)/self.J)

        shared_separator = []
        shared_hidden = _hidden_layers(K_shared,self.topology,self.activation,self.regularisation,self.hidden_bias,'shared_hidden')
        shared_layers = []

        for j in range(self.J):
            # Get indexes
            idx = np.where((np.array(X_loc) == (j+1)) & (np.array(X_shared) == 1))[0].tolist()
            
            # Create lambda layer
            shared_separator.append(
                Lambda(lambda x,ii: tf.gather(x,ii,axis=1), output_shape=((len(idx),)),name='shared_separator_' + str(j+1), arguments={'ii':idx})(input_layer))

            # Attach to mirror hidden layer
            shared_layers.append(shared_hidden(shared_separator[j]))

        # Set shared output layer
        shared_v_layer = Dense(1, activation='linear', name='shared_output',use_bias=False)
        shared_outputs = [shared_v_layer(shared_layers[j]) for j in range(self.J)]

        # For any remaining layers, create an ASU-NN
        if np.sum(X_shared) < X.shape[1]:
            asu_separator = []
            asu_hidden_layers = []
            asu_outputs = []
            concatenate_layers = []
            v_layers = []

            for j in range(self.J):
                # Get indexes
                idx = np.where((np.array(X_loc) == (j+1)) & (np.array(X_shared) == 0))[0].tolist()
                
                # Create separator layers
                asu_separator.append(
                    Lambda(lambda x,ii: tf.gather(x,ii,axis=1), output_shape=((len(idx),)),name='asu_separator_' + str(j+1), arguments={'ii':idx})(input_layer))

                # Create hidden layers
                asu_hidden_layers.append(_hidden_layers(len(idx),self.topology,self.activation,self.regularisation,self.hidden_bias,'asu_hidden_' + str(j+1))(asu_separator[j]))

                # Attach to ASU outputs
                asu_outputs.append(Dense(1,activation='linear',name = 'asu_output_' + str(j+1))(asu_hidden_layers[j]))

                # Concatenate shared and ASU outputs
                concatenate_layers.append(Concatenate()([shared_outputs[j],asu_outputs[j]]))

                # Attach v layers
                v_layers.append(Dense(1,activation='linear',use_bias=False,kernel_initializer='ones',trainable=False)(concatenate_layers[j]))

            # Create output layer
            output_layer = Concatenate()(v_layers)
        else:
            output_layer = Concatenate()(shared_outputs)

        # Add bias if needed
        if self.output_bias:
            output_layer = BiasLayer(units=self.J)(output_layer)

        if not self.from_logits:
            output_layer = Softmax()(output_layer)
        
        self.model = Model(inputs=input_layer, outputs=output_layer, name='Mirror-ASU-NN')

        # Compile
        self.model.compile(optimizer=self.optimiser, loss=SparseCategoricalCrossentropy(from_logits=self.from_logits))

        # Set early stopping
        if early_stopping:
            self.cb = [EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience)]
        else:
            self.cb = [EarlyStopping(monitor='loss', min_delta=min_delta, patience=patience)]

        # Fit
        self.history = self.model.fit(
            x=X,y=y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data = validation_sample,
            callbacks=self.cb,
            verbose = verbose)
        
        # Set model as trained
        self.is_trained = True

    def predict_utility(self,X):
        assert self.from_logits, "Cannot compute predicted utility if argument 'from_logits' is not True"
        v = self.model(X,training=False).numpy()
        return v

    def predict_proba(self,X):
        if self.from_logits:
            v = self.predict_utility(X)
            p = np.exp(v)/np.sum(np.exp(v),axis=1,keepdims=True)
        else:
            p = self.model(X,training=False).numpy()
        return p

    def gradient(self, X, scaler = None):
        x_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as t:
            t.watch(x_tensor)
            output = []
            for j in range(self.J):
                output.append(self.model(x_tensor)[:,j])

        # result = output
        gradients = []
        for j in range(self.J):
            gr = t.gradient(output[j], x_tensor).numpy()
            if scaler is not None:
                gr = scaler.transform(gr)
            gradients.append(gr)

        gradients = np.stack(gradients,axis=2)
        return gradients

def _hidden_layers(input_shape,topology,activation,regulariser,hidden_bias,name):
    
    # Define input
    input_layer = Input(shape=input_shape)

    # Attach hidden layers
    n_layers = len(topology)
    hidden_layers = []

    # Loop among layers
    for l in range(n_layers):
        if l == 0:
            hidden_layers.append(Dense(topology[l], activation=activation, kernel_regularizer=regulariser, use_bias=hidden_bias, name='hidden_' + str(l+1))(input_layer))
        else:
            hidden_layers.append(Dense(topology[l], activation=activation, kernel_regularizer=regulariser, use_bias=hidden_bias, name='hidden_' + str(l+1))(hidden_layers[-1]))

    # Build the model of hidden layers
    return Model(inputs = input_layer, outputs = hidden_layers[-1],name=name)

# Log-likelihood function
def ll(y_true,y_pred,eps=1e-15):
    
    alternatives = np.unique(y_true)
    J = len(alternatives)
    J_pred = y_pred.shape[1]

    clipped_y_pred = np.clip(y_pred.astype(np.float64), eps, 1-eps)

    if J > 2 or J_pred > 1:
        y_array = np.zeros((y_true.shape[0],J))

        for j in range(J):
            y_array[y_true == alternatives[j],j] = 1

        ll_n = y_array * np.log(clipped_y_pred)
    else:
        ll_n = y_true * np.log(clipped_y_pred.flatten()) + (1-y_true)*np.log(1-clipped_y_pred.flatten())

    return np.sum(ll_n)

# Bias layer
class BiasLayer(keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )
    
    def call(self, inputs):
        return inputs + self.b
    
# Normalise function
class normaliser_shared:
    def __init__(self,X_shared,shared_locations,a=0,b=1):

        # Define scalars
        self.a = a
        self.b = b

        # Set shared locations
        self.X_shared = X_shared
        self.shared_locations = shared_locations

    def fit(self,X):
        
        # Check if array is Pandas DataFrame, and if so, transform it to a numpy array
        if isinstance(X,DataFrame):
            X = X.to_numpy()
        
        min_array = []
        max_array = []

        # Loop among columns
        for x in range(X.shape[1]):
            if self.X_shared[x] == 1:
                for loc in self.shared_locations:
                    if x in loc:
                        min_array.append(X[:,loc].min())
                        max_array.append(X[:,loc].max())
            else:
                min_array.append(X[:,x].min())        
                max_array.append(X[:,x].max())        

        self.min_array = np.array(min_array)
        self.max_array = np.array(max_array)

    def transform(self,X):
        
        # Check if array is Pandas DataFrame, and if so, transform it to a numpy array
        if isinstance(X,DataFrame):
            X = X.to_numpy()

        # Transform data
        X_transformed = self.a + (X-self.min_array)*(self.b - self.a)/(self.max_array - self.min_array)

        # Return transformed array
        return X_transformed