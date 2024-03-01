import random

def objective_function(x1, x2):
    """Calculates the objective function value."""
    return (x1**2 + x2 - 11) + (x1 + x2**2 - 7)**2

def decode_individual(chromosome):
    """Decodes a binary string to real values."""
    x1 = int(chromosome[:10], 2) / 1023 * 6
    x2 = int(chromosome[10:], 2) / 1023 * 6
    return x1, x2

def fitness_function(individual):
    """Calculates the fitness based on objective function value."""
    x1, x2 = decode_individual(individual)
    return 1 / (1 + objective_function(x1, x2))


def selection(population, fitnesses):
    """Selects individuals for reproduction based on stochastic sampling without replacement."""
    # Add a small value to prevent division by zero
    fitnesses = [f + 1e-6 for f in fitnesses]
    probabilities = [f / sum(fitnesses) for f in fitnesses]
    selected = random.choices(population, weights=probabilities, k=len(population))
    return selected


def crossover(individual1, individual2):
    """Performs crossover on two individuals with a probability."""
    if random.random() < 0.8:
        crossover_point = random.randint(1, len(individual1) - 1)
        offspring1 = individual1[:crossover_point] + individual2[crossover_point:]
        offspring2 = individual2[:crossover_point] + individual1[crossover_point:]
        return offspring1, offspring2
    else:
        return individual1, individual2

def mutation(individual):
    """Mutates an individual with a probability."""
    mutated_individual = list(individual)
    for i in range(len(individual)):
        if random.random() < 0.05:
            mutated_individual[i] = "1" if individual[i] == "0" else "0"
    return "".join(mutated_individual)

def genetic_algorithm(population_size=20, num_generations=1000):
    """Implements the genetic algorithm."""
    print('-' * 150)
    while True:
        population = ["".join(random.choices(["0", "1"], k=20)) for _ in range(population_size)]
        best_individual, best_fitness, generation = None, None, None
        for generation in range(num_generations):
            fitnesses = [fitness_function(individual) for individual in population]
            selected_population = selection(population, fitnesses)
            offspring = []
            for i in range(0, len(selected_population), 2):
                parent1, parent2 = selected_population[i], selected_population[i+1]
                offspring1, offspring2 = crossover(parent1, parent2)
                offspring.extend([mutation(offspring1), mutation(offspring2)])
            population_with_elitism = [population[i] for i in range(2)]  # Elitism with top 2 individuals
            population = population_with_elitism + offspring[:population_size - 2]

            individual = max(population, key=fitness_function)
            fitness = fitness_function(individual)
            if best_individual is None:
                best_individual = individual
                best_fitness = fitness
                generation = generation + 1
            elif fitness < best_fitness:
                best_individual = individual
                best_fitness = fitness
                generation = generation + 1
        decoded_tuple = decode_individual(best_individual)
        print(f"Generation {generation+1}\nBest solution: {best_individual}\n Fitness: {best_fitness}\nDecoded individuals: {decoded_tuple}\nFunction: {objective_function(decoded_tuple[0], decoded_tuple[1])}")
        print('-' * 150)
        try:
            res = input('Proceed? [y/n] ')
            print('-' * 150)
        except:
            break
        else:
            if 'n' in res.lower():
                break

if __name__ == "__main__":
    genetic_algorithm()
