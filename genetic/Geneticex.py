import numpy as np
from geneticalgorithm import geneticalgorithm as ga

# Define the objective function to maximize
def fitness_function(X):
    x, y = X
    return -(x * y - x**2 + y)  # Minimize the negative to maximize

# Define variable bounds [(min, max) for x, (min, max) for y]
varbound = np.array([[0, 31], [0, 31]])

# Genetic Algorithm Parameters
algorithm_param = {
    'max_num_iteration': 100,
    'population_size': 10,
    'mutation_probability': 0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.7,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}

# Run Genetic Algorithm
model = ga(function=fitness_function, dimension=2, variable_type='int', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
model.run()

# Get best solution
best_x, best_y = model.output_dict['variable']
best_fitness = -fitness_function([best_x, best_y])  # Convert back to positive

best_x, best_y, best_fitness
