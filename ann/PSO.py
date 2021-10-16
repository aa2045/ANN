# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 23:29:15 2020

@author: Aneesah Abdul Kadhar, Dilani Maheswaran

"""

from helper import Helper
import random
import matplotlib.pyplot as plt
from ANN import ANN


"""
This class defines a Particle that belongs to a swarm
"""
class Particle:

    """
    Initialize Particle

    Parameters:
        position : position of particles in the search space
        best_position : best position of the particle
        velocity : Velocity of the particle in the search space
        fit : fitness of the particle
        best_fitness : best fitness of the particle 

    """
    def __init__(self, position, velocity):
        self.position, self.velocity = position, velocity
        #initialize neural network for the particle
        ann = ANN.initialize_network(self.position)

        #calculate fitness for the ann
        self.fit = ANN.mean_square_error(ann)

        #update initial best_position and best_fitness
        self.best_position, self.best_fittness = self.position, self.fit
        

    def set_position(self, position):
        self.position = position
        
        particle_min = -7.00
        particle_max = 7.00

        if not any(p < particle_min for p in position) and not any(p > particle_max for p in position):
            ann = ANN.initialize_network(self.position)
            #get fitness of the position
            fitness = ANN.mean_square_error(ann)

            #if new fitness is better, update particles best_position and best_fitness
            if fitness < self.best_fittness:
                self.fit = fitness
                self.best_fittness = self.fit
                self.best_position = self.position

    def set_velocity(self, velocity):
        self.velocity = velocity

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_best_pos(self):
        return self.best_position

    def get_fitness(self):
        return self.fit

"""
This class defines swarm optimization
"""
class PSO:
    MSE = []
    W = 0.7
    C_1 = 1.5
    C_2 = 1.2
    train = []
    test = []
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    HIDDEN_SIZE = []
    DIMENSIONS = 0
    SWARM_SIZE = 25
    iterations = 100
    
    approx = ""
    
    def __init__(self):
        #fetch ANN architecture input from user
        PSO.HIDDEN_SIZE = Helper.get_layers()

        #update the hidden size
        ANN.set_hyper_parameters(PSO.HIDDEN_SIZE)
        
    """
    Train and optimize Linear function approximation
    """
    def run_linear(self):
        PSO.INPUT_SIZE = 1
        ANN.INPUT_SIZE = 1
        
        PSO.approx = "linear"
        PSO.train, PSO.test =  Helper.read_file(filename = "Data/1in_linear.txt")
        self.run_pso()
    
    """
    Train and optimize Sine function approximation
    """ 
    def run_sine(self):
        PSO.INPUT_SIZE = 1
        ANN.INPUT_SIZE = 1
        
        PSO.approx = "sine"
        PSO.train, PSO.test =  Helper.read_file(filename = "Data/1in_sine.txt")
        self.run_pso()
    
    """
    Train and optimize cubic function approximation
    """ 
    def run_cubic(self):
        PSO.INPUT_SIZE = 1
        ANN.INPUT_SIZE = 1
        
        PSO.approx = "cubic"
        PSO.train, PSO.test =  Helper.read_file(filename = "Data/1in_cubic.txt")
        self.run_pso()

    """
    Train and optimize tanh function approximation
    """         
    def run_tanh(self):
        PSO.INPUT_SIZE = 1
        ANN.INPUT_SIZE = 1
        
        PSO.approx = "tanh"
        PSO.train, PSO.test =  Helper.read_file(filename = "Data/1in_tanh.txt")
        self.run_pso()
        
    """
    Train and optimize complex function approximation
    """
    def run_complex(self):
        PSO.INPUT_SIZE = 2
        ANN.INPUT_SIZE = 2
        
        PSO.approx = "complex"
        PSO.train, PSO.test =  Helper.read_file(filename = "Data/2in_complex.txt")
        self.run_pso()

    """
    Train and optimize XOR function approximation
    """        
    def run_xor(self):
        PSO.INPUT_SIZE = 2
        ANN.INPUT_SIZE = 2
        
        PSO.approx = "xor"
        PSO.train, PSO.test =  Helper.read_file(filename = "Data/2in_xor.txt")
        self.run_pso()
    
    """
    Run PSO functio
    """
    def run_pso(self):
        ANN.set_data_set(PSO.train, PSO.test)
        self.MSE = []


        global_best = self.optimize()
        global_best_position = global_best[1]         
        self.plot_performance_data(global_best_position)
        self.plot_mse_data()
    
    """
    Fetch dimension of PSO based on ANN parameters
    """
    def get_dimensions(self):
        input_data = PSO.INPUT_SIZE + 1
        dimensions = 0
        for h in PSO.HIDDEN_SIZE:
            dimensions += input_data * h
            input_data = h + 1
        dimensions += input_data * PSO.OUTPUT_SIZE
        return dimensions   
   
    """
    Initialize and return the swarm
    """
    def get_swarm(self):
        swarm = []
        for _ in range(PSO.SWARM_SIZE):
            position = [random.uniform(-1.00, 1.00) for _ in range(PSO.DIMENSIONS)]
            velocity = [0 for _ in range(PSO.DIMENSIONS)]
            particle = Particle(position, velocity)
            swarm.append(particle)
        return swarm
   
    """
    Find the global best of swarm.
    """
    def get_global_best(self, swarm):
       best_fittness = swarm[0].get_fitness()
       best_position = swarm[0].get_position()
       
       #loop through all particles in the swarm, and find the global best
       for particle in swarm:
           if particle.get_fitness() < best_fittness:
               best_fittness = particle.get_fitness()
               best_position = particle.get_position()
       return best_fittness, best_position 
  
    """
    Move particles of the swarm
    """
    def particle_trajectory(self,swarm):
       global_best = self.get_global_best(swarm)
       for particle in swarm:
           new_pos = [0 for _ in range(PSO.DIMENSIONS)]
           new_vel = [0 for _ in range(PSO.DIMENSIONS)]
           for d in range(PSO.DIMENSIONS):

               # fetch multiplicand
               r_1 = random.uniform(0.00, 1.00)
               r_2 = random.uniform(0.00, 1.00)
               
               #below is the formula for findig the new velocity of the particle
               weight = PSO.W * particle.get_velocity()[d]
               cognitive = (PSO.C_1 * r_1) * (particle.get_best_pos()[d] - particle.get_position()[d])
               social = (PSO.C_2 * r_2) * (global_best[1][d] - particle.get_position()[d])
               new_vel[d] = weight + cognitive + social
               
               #calculate new position
               new_pos[d] = particle.get_position()[d] + new_vel[d]
               particle.set_position(new_pos)
               particle.set_velocity(new_vel)
    
    """
    Optimize the particles
    """
    def optimize(self): 
        #fetch the dimensions of the swarm
        PSO.DIMENSIONS = self.get_dimensions() 
        swarm = self.get_swarm()
        
        #for each iteration, move the particle toward best position and find the global best
        for e in range(1, self.iterations+1):
           swarm_best = self.get_global_best(swarm)
           self.MSE.append(swarm_best[0])
           self.particle_trajectory(swarm)
        return self.get_global_best(swarm)
    
    """
    Plot actual vs accuracy chart
    """
    def plot_performance_data(self, particle):
        network = ANN.initialize_network(particle)
        output = []
        target = []
        
        #find the output for test data
        for data in PSO.test:
            output.append(ANN.feed_forward(network, data)[0])
            target.append(data[-1])

        x = range(0, len(PSO.test))
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Test Data')
        ax1.set_ylabel('Data', color='blue')
        line2, = ax1.plot(x, output, '-', c='green', lw='1', label='Actual')
        line3, = ax1.plot(x, target, ':', c='blue', lw='1', label='Target')
        fig.tight_layout()
        fig.legend(loc='center')
        plt.show()
        plt.clf()

    """
    Plot Mean Squared Error chart
    """
    def plot_mse_data(self):
        x = range(0, PSO.iterations)
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE', color='blue')
        line, = ax1.plot(x, self.MSE, '-', c='blue', lw='1', label='MSE')
        fig.tight_layout()
        fig.legend(loc='center')
        plt.show()
        plt.clf()

    
if __name__ == '__main__':
    
    activation_function = Helper.get_activation_function()
    ANN.set_activation_function(activation_function)
    pso = PSO()

    print("Approximating for Linear function....")
    pso.run_linear()
    
    print("Approximating for Sine function....")
    pso.run_sine()
    
    print("Approximating for Complex function....")
    pso.run_complex()    
    
    print("Approximating for Cubic function....")
    pso.run_cubic()    
    
    print("Approximating for Tanh function....")
    pso.run_tanh()    
    
    print("Approximating for XOR function....")
    pso.run_xor()
    