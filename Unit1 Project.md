# Unit 1 Project
## Group 3 - Alex Ramirez, Jackie Gore, Eva Duvaris, Steven Vacha, Ethan Chang


## Part 2 - Neuron Models
***LIF Neuron Model:***


```python
import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, tau_m, tau_ref, v_rest, v_thresh):
        # Neuron parameters
        self.tau_m = tau_m  # Membrane time constant
        self.tau_ref = tau_ref  # Refractory period
        self.v_rest = v_rest  # Resting potential
        self.v_thresh = v_thresh  # Threshold potential

        # Neuron state variables
        self.membrane_potential = v_rest  # Initial membrane potential
        self.refractory_time = 0  # Initial refractory time

    def update(self, dt, current_input):
        # Check if the neuron is in a refractory period
        if self.refractory_time > 0:
            # Neuron is in refractory period, decrement refractory time
            self.refractory_time -= dt
            if self.refractory_time <= 0:
                # Refractory period over, reset membrane potential to resting potential
                self.membrane_potential = self.v_rest
        else:
            # Update membrane potential using leaky integration
            dv = (-(self.membrane_potential - self.v_rest) + current_input) / self.tau_m * dt
            self.membrane_potential += dv

            # Check for threshold crossing
            if self.membrane_potential >= self.v_thresh:
                # Neuron has fired, reset membrane potential to resting potential
                self.membrane_potential = self.v_rest
                # Set refractory period
                self.refractory_time = self.tau_ref

        # Return the updated membrane potential
        return self.membrane_potential

# Simulation parameters
tau_m = 10.0  # Membrane time constant (ms)
tau_ref = 2.0  # Refractory period (ms)
v_rest = -70.0  # Resting potential (mV)
v_thresh = -55.0  # Threshold potential (mV)
dt = 1.0  # Time step (ms)
sim_time = 100  # Simulation time (ms)

# Create LIF neuron
neuron = LIFNeuron(tau_m, tau_ref, v_rest, v_thresh)

# Simulation loop
time_points = np.arange(0, sim_time, dt)
membrane_potentials = []

for t in time_points:
    # Inject a constant input current for demonstration purposes
    input_current = 10.0 if t < 50 or 70 < t < 90 else 0.0

    # Update neuron and store membrane potential
    membrane_potential = neuron.update(dt, input_current)
    membrane_potentials.append(membrane_potential)

# Plot results
plt.plot(time_points, membrane_potentials)
plt.axhline(y=-60, color='r', linestyle='--')  # Add the horizontal red dotted line at -60 mV
plt.title('Leaky Integrate-and-Fire Neuron')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.show()

```


![image](https://github.com/aramii3/NNB-Project-Unit1/assets/156377831/8d9022d9-a87e-4fb0-a9e4-7cab413ae7fd)
    


***a LIF model with voltage-gated sodium channel neuron***


This model shows a simple electric circuit where the neurons "battery" has a resting voltage that changes when it recieves a signal aka a current. When this reaches a certain threshold it triggers the neuron to fire an electrical pulse aka an action potential then it resets back to it's resting voltage. In this specific neuron there is a channel called a voltage gated sodium channel that opens when the membrane potential meets a certain threshold, once the gate channel is open it allos sodium ions to flood in and spike membrane potential. The benefits of this model are in it's simplicity and ease to understand, it is also good for large networks of neuron models and long-term models. It also has a good deal of real biological basis and is good for modeling real neurons. However, its limitations come in the fact that it is too simple to truly capture the complexitity of biological neurons and it cannot model everything a biological neuron would do or things like adaptations. This model aslo does not account for percise spike timing and it is very sensitive to it's parameters such as the time constant. This model is simple, easy to make and understand but it is not very complex and cannot hold the same about of bits of information as a more complex model. 


```python
import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, tau_m, tau_ref, v_rest, v_thresh, sodium_channel_params):
        # Neuron parameters
        self.tau_m = tau_m  # Membrane time constant
        self.tau_ref = tau_ref  # Refractory period
        self.v_rest = v_rest  # Resting potential
        self.v_thresh = v_thresh  # Threshold potential

        # Sodium channel parameters
        self.sodium_channel_params = sodium_channel_params
        self.m = 0.0  # Initial activation variable
        self.h = 1.0  # Initial inactivation variable

        # Neuron state variables
        self.membrane_potential = v_rest  # Initial membrane potential
        self.refractory_time = 0  # Initial refractory time

    def update_sodium_channel(self, dt):
        # Update sodium channel dynamics using Hodgkin-Huxley model
        alpha_m = self.sodium_channel_params['alpha_m'](self.membrane_potential)
        beta_m = self.sodium_channel_params['beta_m'](self.membrane_potential)
        alpha_h = self.sodium_channel_params['alpha_h'](self.membrane_potential)
        beta_h = self.sodium_channel_params['beta_h'](self.membrane_potential)

        dm_dt = alpha_m * (1 - self.m) - beta_m * self.m
        dh_dt = alpha_h * (1 - self.h) - beta_h * self.h

        self.m += dm_dt * dt
        self.h += dh_dt * dt

    def update(self, dt, current_input):
        # Check if the neuron is in a refractory period
        if self.refractory_time > 0:
            # Neuron is in refractory period, reset membrane potential to resting potential
            self.refractory_time -= dt
            self.membrane_potential = self.v_rest
        else:
            # Update sodium channel dynamics
            self.update_sodium_channel(dt)

            # Update membrane potential using leaky integration and sodium channel contribution
            dv = (-(self.membrane_potential - self.v_rest) + current_input + \
                  self.sodium_channel_params['g_Na'] * self.m**3 * self.h * \
                  (self.sodium_channel_params['E_Na'] - self.membrane_potential)) / self.tau_m * dt

            self.membrane_potential += dv

            # Check for threshold crossing
            if self.membrane_potential >= self.v_thresh:
                # Neuron has fired, reset membrane potential to exactly +30mV
                self.membrane_potential = self.v_rest + 30
                # Set refractory period
                self.refractory_time = self.tau_ref

        # Return the updated membrane potential
        return self.membrane_potential

# Sodium channel parameters (Hodgkin-Huxley model)
sodium_channel_params = {
    'alpha_m': lambda v: 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10)),
    'beta_m': lambda v: 4.0 * np.exp(-(v + 65) / 18),
    'alpha_h': lambda v: 0.07 * np.exp(-(v + 65) / 20),
    'beta_h': lambda v: 1.0 / (1 + np.exp(-(v + 35) / 10)),
    'g_Na': 120.0,  # Sodium conductance (mS/cm^2)
    'E_Na': 50.0,   # Sodium reversal potential (mV)
}

# Simulation parameters
tau_m = 10.0  # Membrane time constant (ms)
tau_ref = 2.0  # Refractory period (ms)
v_rest = -70.0  # Resting potential (mV)
v_thresh = -55.0  # Threshold potential (mV)
dt = 0.1  # Time step (ms)
sim_time = 100  # Simulation time (ms)

# Create LIF neuron with sodium channels
neuron = LIFNeuron(tau_m, tau_ref, v_rest, v_thresh, sodium_channel_params)

# Simulation loop
time_points = np.arange(0, sim_time, dt)
membrane_potentials = []

for t in time_points:
    # Inject a step input current for demonstration purposes
    input_current = 10.0 if 20 < t < 100 else 0.0

    # Update neuron and store membrane potential
    membrane_potential = neuron.update(dt, input_current)
    membrane_potentials.append(membrane_potential)

# Plot results
plt.plot(time_points, membrane_potentials)
plt.axhline(y=v_thresh, color='r', linestyle='--')  # Add a dotted line across the threshold
plt.title('Leaky Integrate-and-Fire Neuron with Sodium Channel (Multiple Spikes)')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.show()

```


    
![png](output_LIFSO_.png)
    


***Simple Neural Network Model - Open Loop Model***


```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network with given input, hidden, and output layer sizes.
        
        Parameters:
        - input_size (int): Number of input neurons.
        - hidden_size (int): Number of neurons in the hidden layer.
        - output_size (int): Number of output neurons.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly with mean 0
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
    def forward(self, inputs):
        """
        Perform forward pass through the network.
        
        Parameters:
        - inputs (ndarray): Input data of shape (num_samples, input_size).
        
        Returns:
        - ndarray: Output of the network of shape (num_samples, output_size).
        """
        # Calculate activations of the hidden layer
        hidden_inputs = np.dot(inputs, self.weights_input_hidden)
        hidden_outputs = self.sigmoid(hidden_inputs)
        
        # Calculate activations of the output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = self.sigmoid(final_inputs)
        
        return final_outputs
    
    def sigmoid(self, x):
        """
        Sigmoid activation function.
        
        Parameters:
        - x (ndarray): Input to the function.
        
        Returns:
        - ndarray: Output after applying sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def visualize(self):
        """
        Visualize the neural network architecture.
        """
        G = nx.DiGraph()
        G.add_nodes_from(['Input {}'.format(i) for i in range(self.input_size)], layer='Input')
        G.add_nodes_from(['Hidden {}'.format(i) for i in range(self.hidden_size)], layer='Hidden')
        G.add_nodes_from(['Output {}'.format(i) for i in range(self.output_size)], layer='Output')
        
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                G.add_edge('Input {}'.format(i), 'Hidden {}'.format(j), weight=self.weights_input_hidden[i, j])
        
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                G.add_edge('Hidden {}'.format(i), 'Output {}'.format(j), weight=self.weights_hidden_output[i, j])
        
        pos = nx.multipartite_layout(G, subset_key="layer")
        
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, arrowsize=20)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Neural Network Architecture")
        plt.show()

# Example usage
input_size = 2
hidden_size = 3
output_size = 1

# Create a neural network with 2 input neurons, 3 hidden neurons, and 1 output neuron
model = NeuralNetwork(input_size, hidden_size, output_size)

# Visualize the network architecture
model.visualize()

```


    
![png](output_6_0.png)
    



```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network with given input, hidden, and output layer sizes.
        
        Parameters:
        - input_size (int): Number of input neurons.
        - hidden_size (int): Number of neurons in the hidden layer.
        - output_size (int): Number of output neurons.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly with mean 0
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
    def forward(self, inputs):
        """
        Perform forward pass through the network.
        
        Parameters:
        - inputs (ndarray): Input data of shape (num_samples, input_size).
        
        Returns:
        - ndarray: Output of the network of shape (num_samples, output_size).
        """
        # Calculate activations of the hidden layer
        hidden_inputs = np.dot(inputs, self.weights_input_hidden)
        hidden_outputs = self.sigmoid(hidden_inputs)
        
        # Calculate activations of the output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = self.sigmoid(final_inputs)
        
        return final_outputs
    
    def sigmoid(self, x):
        """
        Sigmoid activation function.
        
        Parameters:
        - x (ndarray): Input to the function.
        
        Returns:
        - ndarray: Output after applying sigmoid function.
        """
        return 1 / (1 + np.exp(-x))
    
    def visualize(self):
        """
        Visualize the neural network architecture using networkx and matplotlib.
        """
        G = nx.DiGraph()

        # Add nodes for each layer
        G.add_nodes_from(range(self.input_size), layer='Input')
        G.add_nodes_from(range(self.input_size, self.input_size + self.hidden_size), layer='Hidden')
        G.add_nodes_from(range(self.input_size + self.hidden_size, self.input_size + self.hidden_size + self.output_size), layer='Output')

        # Add edges between layers
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                G.add_edge(i, self.input_size + j, weight=self.weights_input_hidden[i, j])
        
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                G.add_edge(self.input_size + i, self.input_size + self.hidden_size + j, weight=self.weights_hidden_output[i, j])
        
        # Specify the node positions
        pos = nx.multipartite_layout(G, subset_key="layer")

        # Draw the neural network
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=12, arrowsize=20)
        edge_labels = {(i, j): '{:.2f}'.format(d['weight']) for i, j, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
        plt.title("Neural Network Architecture")
        plt.show()

# Example usage
input_size = 2
hidden_size = 3
output_size = 1

# Create a neural network with 2 input neurons, 3 hidden neurons, and 1 output neuron
model = NeuralNetwork(input_size, hidden_size, output_size)

# Visualize the network architecture
model.visualize()


```

    
![png](output_7_0.png)
    

## Part 3 - Model Descriptions/Comparisons


An LIF model (Leaky integrate and fire model) is a model in which the behavior of a neuron is simulated as a measure of membrane potential over time. The benefits include its simplicity in implementation when attempting to model biological neuron behavior. Though simple in design, the LIF model can often oversimplify biological neuronal function and may fail to account for changes in threshold, synaptic plasticity, etc.


An LIF model with a voltage-gated sodium channel factored in operates in the same way, but more accurately models the workings of an action potential with the rapid depolarization and refractory model. As a result, benefits of using the model include its ability to better model an action potential including the cell's ability to excite and fire. A major limitation, however, is its increased complexity and strain on computing power. In addition, it can be slow and inefficient to compute large-scale networks, as it involves more information than a simple LIF model.


Simple neural networks are computational models that are more concerned with the interconnectedness of neurons in the brain and utilize nodes organized into layers to compute data. Benefits of using this model include its ability to take in a variety of different data depending on how many layers the model has while being able to process more than one stream of information at once through parallel processing. Limitations of such a model include difficulties interpreting the hidden layer in between the input and output in addition to the large amounts of data that are needed to train the model to compute effectively. Artificial neural networks are more functional for utility in machine learning and natural language processing than they are for biological realism.


Overall, simple neural networks are the most complex and contain the most bits of information. Simple neural networks can contain multiple hidden layers that better represent the complexity of the factors that contribute to neural connections between 2 cells. Even if LIF models contain voltage-gated sodium channels to better represent the transmission of action potentials and depolarization of the cell to reach threshold, it still doesn't model other factors that can contribute to the communication of information between cells such as summation of graded potentials, threshold changes, etc. Furthermore, simple neural networks are ANNs and DNN that allow for feature detection, which allows it to learn from its own programming while processing vast amounts of information. In comparison, both LIF models do not reach the complexity of ANN and cannot process or adapt to the vast amounts of information that may be inputted into the model.


The LIF model without voltage-gated sodium channels is more concise as it is the least complex. LIF models simply integrate signals and produce an action potential spike when threshold is met. Compared to the other two models that are able to integrate more information and better model the behavior of neural connections, the simple LIF model is the most concise as its computational capabilities are the simplest.



## Part 4 - Varying Given Inputs to Updated LIF Model

***What if inputs to LIF model come in bursting patterns?***

```python
import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, tau_m, tau_ref, v_rest, v_thresh, sodium_channel_params):
        # Neuron parameters
        self.tau_m = tau_m  # Membrane time constant
        self.tau_ref = tau_ref  # Refractory period
        self.v_rest = v_rest  # Resting potential
        self.v_thresh = v_thresh  # Threshold potential

        # Sodium channel parameters
        self.sodium_channel_params = sodium_channel_params
        self.m = 0.0  # Initial activation variable
        self.h = 1.0  # Initial inactivation variable

        # Neuron state variables
        self.membrane_potential = v_rest  # Initial membrane potential
        self.refractory_time = 0  # Initial refractory time

    def update_sodium_channel(self, dt):
        # Update sodium channel dynamics using Hodgkin-Huxley model
        alpha_m = self.sodium_channel_params['alpha_m'](self.membrane_potential)
        beta_m = self.sodium_channel_params['beta_m'](self.membrane_potential)
        alpha_h = self.sodium_channel_params['alpha_h'](self.membrane_potential)
        beta_h = self.sodium_channel_params['beta_h'](self.membrane_potential)

        dm_dt = alpha_m * (1 - self.m) - beta_m * self.m
        dh_dt = alpha_h * (1 - self.h) - beta_h * self.h

        self.m += dm_dt * dt
        self.h += dh_dt * dt

    def update(self, dt, current_input):
        # Check if the neuron is in a refractory period
        if self.refractory_time > 0:
            # Neuron is in refractory period, reset membrane potential to resting potential
            self.refractory_time -= dt
            self.membrane_potential = self.v_rest
        else:
            # Update sodium channel dynamics
            self.update_sodium_channel(dt)

            # Update membrane potential using leaky integration and sodium channel contribution
            dv = (-(self.membrane_potential - self.v_rest) + current_input + \
                  self.sodium_channel_params['g_Na'] * self.m**3 * self.h * \
                  (self.sodium_channel_params['E_Na'] - self.membrane_potential)) / self.tau_m * dt

            self.membrane_potential += dv

            # Check for threshold crossing
            if self.membrane_potential >= self.v_thresh:
                # Neuron has fired, reset membrane potential to resting potential
                self.membrane_potential = self.v_rest
                # Set refractory period
                self.refractory_time = self.tau_ref

        # Return the updated membrane potential
        return self.membrane_potential

# Sodium channel parameters (Hodgkin-Huxley model)
sodium_channel_params = {
    'alpha_m': lambda v: 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10)),
    'beta_m': lambda v: 4.0 * np.exp(-(v + 65) / 18),
    'alpha_h': lambda v: 0.07 * np.exp(-(v + 65) / 20),
    'beta_h': lambda v: 1.0 / (1 + np.exp(-(v + 35) / 10)),
    'g_Na': 120.0,  # Sodium conductance (mS/cm^2)
    'E_Na': 50.0,   # Sodium reversal potential (mV)
}

# Simulation parameters
tau_m = 10.0  # Membrane time constant (ms)
tau_ref = 2.0  # Refractory period (ms)
v_rest = -70.0  # Resting potential (mV)
v_thresh = -55.0  # Threshold potential (mV)
dt = 0.1  # Time step (ms)
sim_time = 1000  # Simulation time (ms)
burst_period = 200  # Burst period (ms)
quiescent_period = 100  # Quiescent period (ms)

# Create LIF neuron with sodium channels for bursting pattern
bursting_neuron = LIFNeuron(tau_m, tau_ref, v_rest, v_thresh, sodium_channel_params)

# Create LIF neuron with sodium channels for normal firing pattern
normal_neuron = LIFNeuron(tau_m, tau_ref, v_rest, v_thresh, sodium_channel_params)

# Simulation loop
time_points = np.arange(0, sim_time, dt)
bursting_membrane_potentials = []
normal_membrane_potentials = []

for t in time_points:
    # Determine input current based on bursting or quiescent period
    if t % (burst_period + quiescent_period) < burst_period:
        bursting_input_current = 10.0  # High input current during bursting period
    else:
        bursting_input_current = 0.0  # No input current during quiescent period

    # Update bursting neuron and store membrane potential
    bursting_membrane_potential = bursting_neuron.update(dt, bursting_input_current)
    bursting_membrane_potentials.append(bursting_membrane_potential)

    # Update normal firing neuron with constant input current
    normal_input_current = 10.0  # Constant input current for normal firing
    normal_membrane_potential = normal_neuron.update(dt, normal_input_current)
    normal_membrane_potentials.append(normal_membrane_potential)


# Plot results for bursting neuron
plt.plot(time_points, bursting_membrane_potentials, label='Bursting Neuron', color='blue')
plt.axhline(y=-55, color='r', linestyle='--')  # Add the horizontal red dotted line at -55 mV
plt.title('Leaky Integrate-and-Fire Neurons with Sodium Channels (Bursting Pattern)')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.legend()
plt.show()

# Plot results for normal firing neuron
plt.plot(time_points, normal_membrane_potentials, label='Normal Firing Neuron', color='black', linestyle='--')
plt.axhline(y=-55, color='r', linestyle='--')  # Add the horizontal red dotted line at -55 mV
plt.title('Leaky Integrate-and-Fire Neurons with Sodium Channels (Normal Firing Pattern)')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.legend()
plt.show()


```

![png](output_burst_.png)
![png](output_normal_.png)



In this example we changed the firing pattern of the presynaptic inputs so that the postsynaptic neuron receives a bursting pattern of inputs. Bursts of presynaptic activity can lead to varying patterns of postsynaptic responses depending on their frequency and timing. In the graph above, the normal firing pattern (in dotted black) is a consistent tonic pattern without any cyclical trends. In the updated system (blue lines) there is a bursting period of 200ms, where the neuron fires in a rapid and tonic pattern, followed by a 100ms quiescent period, where firing is paused while retaining the potential to resume firing upon stimuli. This creates a cyclic bursting firing pattern of the postsynaptic neuron. In terms of signal integration, bursting inputs may cause more pronounced depolarization and trigger action potentials more readily, which leads to altered postsynaptic firing patterns.



***What happens if graded potentials don't sum to threshold?***


```python
#Graded potentials that don't sum to threshold
import numpy as np
import matplotlib.pyplot as plt

class LIFNeuron:
    def __init__(self, tau_m, tau_ref, v_rest, v_thresh, sodium_channel_params):
        # Neuron parameters
        self.tau_m = tau_m  # Membrane time constant
        self.tau_ref = tau_ref  # Refractory period
        self.v_rest = v_rest  # Resting potential
        self.v_thresh = v_thresh  # Threshold potential

        # Sodium channel parameters
        self.sodium_channel_params = sodium_channel_params
        self.m = 0.0  # Initial activation variable
        self.h = 1.0  # Initial inactivation variable

        # Neuron state variables
        self.membrane_potential = v_rest  # Initial membrane potential
        self.refractory_time = 0  # Initial refractory time

    def update_sodium_channel(self, dt):
        # Update sodium channel dynamics using Hodgkin-Huxley model
        alpha_m = self.sodium_channel_params['alpha_m'](self.membrane_potential)
        beta_m = self.sodium_channel_params['beta_m'](self.membrane_potential)
        alpha_h = self.sodium_channel_params['alpha_h'](self.membrane_potential)
        beta_h = self.sodium_channel_params['beta_h'](self.membrane_potential)

        dm_dt = alpha_m * (1 - self.m) - beta_m * self.m
        dh_dt = alpha_h * (1 - self.h) - beta_h * self.h

        self.m += dm_dt * dt
        self.h += dh_dt * dt

    def update(self, dt, current_input):
        # Check if the neuron is in a refractory period
        if self.refractory_time > 0:
            # Neuron is in refractory period, reset membrane potential to resting potential
            self.refractory_time -= dt
            self.membrane_potential = self.v_rest
        else:
            # Update sodium channel dynamics
            self.update_sodium_channel(dt)

            # Update membrane potential using leaky integration and sodium channel contribution
            dv = (-(self.membrane_potential - self.v_rest) + current_input + \
                  self.sodium_channel_params['g_Na'] * self.m**3 * self.h * \
                  (self.sodium_channel_params['E_Na'] - self.membrane_potential)) / self.tau_m * dt

            self.membrane_potential += dv

            # Check for threshold crossing
            if self.membrane_potential >= self.v_thresh:
                # Neuron has fired, reset membrane potential to resting potential
                self.membrane_potential = self.v_rest
                # Set refractory period
                self.refractory_time = self.tau_ref

        # Return the updated membrane potential
        return self.membrane_potential

# Sodium channel parameters (Hodgkin-Huxley model)
sodium_channel_params = {
    'alpha_m': lambda v: 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10)),
    'beta_m': lambda v: 4.0 * np.exp(-(v + 65) / 18),
    'alpha_h': lambda v: 0.07 * np.exp(-(v + 65) / 20),
    'beta_h': lambda v: 1.0 / (1 + np.exp(-(v + 35) / 10)),
    'g_Na': 120.0,  # Sodium conductance (mS/cm^2)
    'E_Na': 50.0,   # Sodium reversal potential (mV)
}

# Simulation parameters
tau_m = 10.0  # Membrane time constant (ms)
tau_ref = 2.0  # Refractory period (ms)
v_rest = -75.0  # Resting potential (mV) - hyperpolarization
v_thresh = -55.0  # Threshold potential (mV)
dt = 0.1  # Time step (ms)
sim_time = 800  # Simulation time (ms)

# Create LIF neuron with sodium channels
neuron = LIFNeuron(tau_m, tau_ref, v_rest, v_thresh, sodium_channel_params)

# Simulation loop
time_points = np.arange(0, sim_time, dt)
membrane_potentials = []

for t in time_points:
    # Inject input current with different amplitudes
    if 20 <= t < 250:
        input_current = np.random.choice([5.0, 10.0])  # Random input currents below and above threshold
    elif 250 <= t < 300:
        input_current = 5.0  # Graded potential below threshold
    elif 300 <= t < 350:
        input_current = 10.0  # Step input current to trigger spike
    elif 750 <= t < 800:
        input_current = 10.0  # Step input current to trigger spike
    else:
        input_current = 0.0  # No input current

    # Update neuron and store membrane potential
    membrane_potential = neuron.update(dt, input_current)
    membrane_potentials.append(membrane_potential)

# Plot results
plt.plot(time_points, membrane_potentials)
plt.axhline(y=-62.5, color='r', linestyle='--')  # Add the horizontal dotted line at -62.5 mV
plt.title('Leaky Integrate-and-Fire Neuron with Sodium Channel (Hyperpolarization at Beginning)')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.show()

# Define parameters for the regular firing neuron
regular_firing_params = {
    'tau_m': 10.0,  # Membrane time constant (ms)
    'tau_ref': 2.0,  # Refractory period (ms)
    'v_rest': -70.0,  # Resting potential (mV)
    'v_thresh': -55.0,  # Threshold potential (mV)
    'sodium_channel_params': sodium_channel_params
}

# Create regular firing neuron
regular_firing_neuron = LIFNeuron(**regular_firing_params)

# Simulation loop for regular firing neuron
regular_firing_membrane_potentials = []

for t in time_points:
    # Apply constant input current
    input_current = 10.0
    membrane_potential = regular_firing_neuron.update(dt, input_current)
    regular_firing_membrane_potentials.append(membrane_potential)

# Plot results
plt.plot(time_points, membrane_potentials, label='Bursting Neuron')  # Plot bursting neuron
plt.plot(time_points, regular_firing_membrane_potentials, label='Regular Firing Neuron', color='green')  # Plot regular firing neuron
plt.axhline(y=-62.5, color='r', linestyle='--')  # Add the horizontal dotted line at -62.5 mV

```

![image](https://github.com/aramii3/NNB-Project-Unit1/assets/156377831/01ebe4a2-3062-47fd-8fed-2e15a304b9fb)
![image](https://github.com/aramii3/NNB-Project-Unit1/assets/156377831/c1f5b736-5f8f-46da-af77-23a2cf13a683)


In the updated LIF model, if the inputs do not sum to reach the firing threshold, the postsynaptic neuron will not generate an action potential/spike. When the postsynaptic neuron integrates its synaptic inputs over time, each input has its own weight, which represents its effectiveness in depolarizing or hyperpolarizing the neuron’s membrane potential. If the integrated effect of the inputs causes the membrane potential to reach/exceed the neuron’s firing threshold, which is typically a fixed value in the LIF model, an action potential is generated. In the model above, the green firing pattern shows a normal LIF pattern where the synaptic inputs add to threshold value and generate rhythmic repetitive firing. The blue firing pattern shows a neuron where synaptic inputs only sum to threshold at around 300-350ms and 750-800ms, where spikes are generated. Elsewhere during synaptic input integration, the inputs were not able to sum to threshold to generate an action potential.


## Part 5 - Inhibitory Inputs ##

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
Cm = 1.0  # Membrane capacitance (uF/cm^2)
gL = 0.1  # Leak conductance (mS/cm^2)
EL = -65  # Leak reversal potential (mV)
ENa = 55  # Sodium reversal potential (mV)
gNa_max = 35  # Maximum sodium conductance (mS/cm^2)
Vt = -55  # Threshold voltage (mV)
Vr = -65  # Reset voltage (mV)
I_exc = 10  # Excitatory input current (nA)
I_inh = 5  # Inhibitory input current (nA)
dt = 0.01  # Time step (ms)
t_max = 50  # Maximum time (ms)

# Initialize variables
t = np.arange(0, t_max + dt, dt)
V_no_inh = np.zeros(len(t))
V_with_inh = np.zeros(len(t))
m = np.zeros(len(t))
h = np.zeros(len(t))
V_no_inh[0] = Vr
V_with_inh[0] = Vr

# Simulate the LIF neuron with voltage-gated sodium channels
for i in range(1, len(t)):
    # Update sodium channel gating variables using Hodgkin-Huxley model
    alpha_m = 0.1 * (V_no_inh[i-1] + 40) / (1 - np.exp(-(V_no_inh[i-1] + 40) / 10))
    beta_m = 4 * np.exp(-(V_no_inh[i-1] + 65) / 18)
    alpha_h = 0.07 * np.exp(-(V_no_inh[i-1] + 65) / 20)
    beta_h = 1 / (1 + np.exp(-(V_no_inh[i-1] + 35) / 10))
    m[i] = m[i-1] + dt * (alpha_m * (1 - m[i-1]) - beta_m * m[i-1])
    h[i] = h[i-1] + dt * (alpha_h * (1 - h[i-1]) - beta_h * h[i-1])

    # Calculate sodium current without inhibitory input
    INa_no_inh = gNa_max * m[i]**3 * h[i] * (V_no_inh[i-1] - ENa)

    # Update membrane potential without inhibitory input
    V_no_inh[i] = V_no_inh[i-1] + dt * ((I_exc - gL * (V_no_inh[i-1] - EL) - INa_no_inh) / Cm)

    # Calculate sodium current with inhibitory input
    INa_with_inh = gNa_max * m[i]**3 * h[i] * (V_with_inh[i-1] - ENa)

    # Update membrane potential with inhibitory input
    V_with_inh[i] = V_with_inh[i-1] + dt * (((I_exc - I_inh) - gL * (V_with_inh[i-1] - EL) - INa_with_inh) / Cm)

    # Check for spike without inhibitory input
    if V_no_inh[i] >= Vt:
        V_no_inh[i] = Vr

    # Check for spike with inhibitory input
    if V_with_inh[i] >= Vt:
        V_with_inh[i] = Vr

# Plot membrane potential without inhibitory input
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t, V_no_inh)
plt.axhline(y=Vt, color='r', linestyle='--')  # Add threshold line
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Without Inhibitory Inputs')

# Plot membrane potential with inhibitory input
plt.subplot(1, 2, 2)
plt.plot(t, V_with_inh)
plt.axhline(y=Vt, color='r', linestyle='--')  # Add threshold line
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('With Inhibitory Inputs')

plt.tight_layout()
plt.show()
```
![png](inhibitoryinput1.png)
In this first graph labeled without inhibitory inputs the LIF neuron is only given excitatory inputs and no inhibitory inputs. You can see the membrane potential increase in response to the excitation until it hits the threshold where it then fires a action potential and then resets and continues on this pattern. However, in the secound graph labeled with inhibitory input the LIF neuron is given both excitatory and inhibitory impluses. The inhibitory impulses hyperpolarize the membrane potential which makes it less likely to reach it's needed threshold voltage. This is what causes the rate of the action potentials the decrease and why you can see that the membrane potential looks different. In this graph the inhibitry input is given continously at every ms so you see that the action potentials have been cut in half. There is a dotted red line that represents the threshold that muct be meet for an action potential to fire at the top of each graph.

## Part 6 - Entropy


***What is Shannon Entropy?***


Shannon Entropy is defined as the amount of uncertainty involved in the value of a random variable or process. It is represented with the variable H using the equation below (retrieved from NEUR3002 In-Class Presentation #6 Slide 21):

![png](entropy.png)

In this equation, P(x) is the probability of the event x occurring. The sum of probabilities is taken over all of the possible events x in the sample space.

Now that we have defined Shannon Entropy, what does it mean in the context of information theory?

In the context of information theory, entropy quantifies the uncertainty associated with a source/system of information. The higher the entropy, the more unpredictable the system is. This uncertainty is a result of the variability in the occurrence of different symbols or characters within the data. It becomes harder to predict the outcome of any event due to the diversity of symbols and characters present within the data. 
Shannon entropy is calculated based on the average distribution of characters or symbols in the data source. Shannon entropy will take into account how often each character or symbol is present in the data. Shorter codes will be assigned to more frequent characters whereas longer codes will be assigned to less frequent characters, thereby achieving optimal efficiency. 

How does this relate to encoding efficiency?

An encoding scheme is a method by which data is represented in a compact form. The efficiency of encoding schemes is determined by how far the data can be compressed while retaining all of its information. Shannon entropy provides a theoretical limit on the efficiency of encoding schemes, which is defined as the variable H. If there was high Shannon entropy within a source of information, there would be greater uncertainty and the encoding scheme would need to use more bits on average per symbol in order to retain all of the information.

Can you give a practical example?

Suppose we have a bag of 10 uniquely colored marbles, each having the same probability of being picked. What is the most efficient way to represent the color of each marble using bits?

Since there are 10 colors, we could potentially use 4 bits to represent all of the colors. 2^4 = 16, which is greater than 10. However, this would be inefficient as we would have 6 unused states (16-10 = 6). 
An alternative solution would be to group the marbles into sets of 3, and then assign bits to represent each set of marbles. With 3 marbles in each group, we will have 10^3 = 1000 possible combinations of colors. In order to represent 1000 combinations, we would need to calculate log2(1000), which is approximately 10 bits. With 10 bits, we are able to efficiently encode the color combination of the marbles. Because we grouped the marbles into sets of 3, each marble can be represented by 10/3 bits, which is approximately 3.333 bits. Ideally, the most efficient way to represent the number of color combinations would be log2(10) (3.32 bits), due to the fact that if we have n equally likely outcomes, the entropy will be log2(n) bits. 






```python
def create_cat():
    cat = r'''
   /\_/\  
  ( o.o ) 
   > ^ <
    '''
    return cat

def say_thank_you():
    cat = create_cat()
    message = "Thank you!"
    lines = [""] * max(cat.count('\n'), message.count('\n'))

    cat_lines = cat.split('\n')
    message_lines = message.split('\n')

    for i in range(len(lines)):
        cat_line = cat_lines[i] if i < len(cat_lines) else ""
        message_line = message_lines[i] if i < len(message_lines) else ""
        lines[i] = f"{cat_line.ljust(15)} {message_line.center(10)}"

    print("\n".join(lines))

say_thank_you()

```


**Cat Fact**: The wealtheist cat in the world was named blackie and inherited 7 million dollars from his owner after he died; however, Taylor Swift's cat has an estimated 97 million in assets!


**Additional Cat Fact**: Cats can't taste sweetness due to a lack of the protein receptor that allows for sweetness detection. They lack on of the proteins in the sweetness channel heteromer protein complex.
