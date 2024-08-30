import numpy as np
import matplotlib.pyplot as plt

# Definimos los parámetros del problema
n_sensors = 10          # Número de sensores que tiene cada agente
region_size = 100       # Tamaño de la región (área) donde se ubicarán los sensores (100x100)
n_agents = 20           # Número de agentes (soluciones) en el algoritmo GSA
n_iterations = 100      # Número de iteraciones que realizará el algoritmo

# Establecemos la posición de la estación base en el centro de la región
base_station = np.array([region_size / 2, region_size / 2])

# Inicialización aleatoria de posiciones de sensores para cada agente
# Cada agente tiene 10 sensores, cada sensor tiene una posición en 2D (x, y)
agents = np.random.rand(n_agents, n_sensors, 2) * region_size

# Función de aptitud (fitness) que evalúa la calidad de cada agente
def fitness(agent):
    coverage_radius = 20  # Radio de cobertura de cada sensor
    energy_consumption = 0
    uncovered_area = 0
    disconnected_penalty = 0

    # Evaluamos cada sensor del agente
    for sensor in agent:
        # Calculamos la distancia del sensor a la estación base
        distance_to_base = np.linalg.norm(sensor - base_station)
        energy_consumption += distance_to_base  # Aumentamos el consumo de energía proporcionalmente a la distancia
        
        # Calculamos la cobertura de área del sensor
        x = np.linspace(0, region_size, 100)
        y = np.linspace(0, region_size, 100)
        xx, yy = np.meshgrid(x, y)
        distances = np.sqrt((xx - sensor[0])**2 + (yy - sensor[1])**2)
        covered = np.sum(distances <= coverage_radius)
        uncovered_area += region_size**2 - covered
        
        # Penalización si el sensor está desconectado (demasiado lejos de la estación base)
        if distance_to_base > coverage_radius:
            disconnected_penalty += 1
            
    # La función de aptitud combina el consumo de energía, el área no cubierta y las desconexiones
    return energy_consumption + uncovered_area + disconnected_penalty * 1000

# Función que calcula la fuerza gravitacional entre dos agentes
def gravitational_force(agent_i, agent_j, mass_j):
    # Calculamos la distancia entre dos agentes
    distance = np.linalg.norm(agent_i - agent_j, axis=1)
    # Calculamos la fuerza gravitacional considerando la masa del segundo agente
    force = mass_j / (distance + 1e-5)[:, np.newaxis]  # Añadimos un pequeño valor para evitar divisiones por cero
    return force

# Función que actualiza la posición de un agente basado en la fuerza gravitacional y la velocidad
def update_position(agent, force, velocity):
    velocity = 0.5 * velocity + force  # Actualizamos la velocidad del agente
    agent += velocity  # Movemos al agente según su nueva velocidad
    agent = np.clip(agent, 0, region_size)  # Limitar las posiciones a los límites de la región
    return agent, velocity

# Inicializamos las velocidades de los agentes como cero (están en reposo al inicio)
velocities = np.zeros_like(agents)

# Ejecución del algoritmo GSA (iteraciones para mejorar la posición de los agentes)
best_fitness = np.inf  # Valor inicial muy alto para representar la peor aptitud posible
best_agent = None      # Variable para almacenar el mejor agente encontrado
fitness_history = []   # Lista para almacenar la evolución del mejor fitness
agent_positions_history = []  # Lista para almacenar las posiciones de los agentes en cada iteración

for iteration in range(n_iterations):
    # Calculamos el valor de fitness para cada agente
    fitness_values = np.array([fitness(agent) for agent in agents])
    
    # Normalizamos las masas de los agentes en base a sus valores de fitness
    masses = (fitness_values.max() - fitness_values) / (fitness_values.max() - fitness_values.min() + 1e-5)
    
    # Calculamos el centroide de cada agente (promedio de las posiciones de sus sensores)
    centroids = np.mean(agents, axis=1)
    agent_positions_history.append(np.copy(centroids))  # Guardamos las posiciones para visualización
    
    # Identificamos al agente con el mejor fitness en esta iteración
    min_idx = np.argmin(fitness_values)
    if fitness_values[min_idx] < best_fitness:
        best_fitness = fitness_values[min_idx]
        best_agent = np.copy(agents[min_idx])  # Guardamos el mejor agente encontrado
    
    fitness_history.append(best_fitness)  # Guardamos el mejor fitness de esta iteración
    
    # Calculamos las fuerzas gravitacionales y actualizamos las posiciones de los agentes
    for i in range(n_agents):
        total_force = np.zeros((n_sensors, 2))
        for j in range(n_agents):
            if i != j:
                total_force += gravitational_force(agents[i], agents[j], masses[j])
        agents[i], velocities[i] = update_position(agents[i], total_force, velocities[i])

# Visualización final de la mejor configuración de sensores encontrada
plt.figure(figsize=(10, 10))
plt.scatter(best_agent[:, 0], best_agent[:, 1], c='blue', label='Sensores')
plt.scatter(base_station[0], base_station[1], c='red', label='Estación base')
plt.xlim(0, region_size)
plt.ylim(0, region_size)
plt.title("Mejor Configuración de Sensores Encontrada por GSA")
plt.legend()
plt.grid(True)
plt.show()

# Graficamos la evolución del fitness a lo largo de las iteraciones
plt.figure(figsize=(8, 6))
plt.plot(fitness_history, label='Mejor Fitness')
plt.xlabel('Iteraciones')
plt.ylabel('Fitness')
plt.title('Evolución del Fitness')
plt.legend()
plt.grid(True)
plt.show()

# Gráficas para iteraciones clave mostrando el movimiento de los agentes (centroides)
iterations_to_plot = [10, 20, 40, 60, 85, 100]

for iter_num in iterations_to_plot:
    plt.figure(figsize=(10, 10))
    agent_centroids = agent_positions_history[iter_num - 1]
    plt.scatter(agent_centroids[:, 0], agent_centroids[:, 1], color='blue', label='Agentes')
    #plt.scatter(base_station[0], base_station[1], color='red', marker='x', s=100, label='Estación Base')
    plt.xlim(0, region_size)
    plt.ylim(0, region_size)
    plt.title(f"Posiciones de los Agentes - Iteración {iter_num}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

