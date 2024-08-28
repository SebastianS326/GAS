import numpy as np
import matplotlib.pyplot as plt

# Definimos los parámetros
n_sensors = 10
region_size = 100
n_agents = 20
n_iterations = 100

# Estación base en el centro de la región
base_station = np.array([region_size / 2, region_size / 2])

# Inicialización aleatoria de posiciones de sensores para cada agente
agents = np.random.rand(n_agents, n_sensors, 2) * region_size

# Función de aptitud que combina cobertura, consumo de energía y conectividad
def fitness(agent):
    coverage_radius = 20  # Radio de cobertura de cada sensor
    energy_consumption = 0
    uncovered_area = 0
    disconnected_penalty = 0

    for sensor in agent:
        distance_to_base = np.linalg.norm(sensor - base_station)
        energy_consumption += distance_to_base  # Consumo de energía proporcional a la distancia
        
        # Calculando la cobertura
        x = np.linspace(0, region_size, 100)
        y = np.linspace(0, region_size, 100)
        xx, yy = np.meshgrid(x, y)
        distances = np.sqrt((xx - sensor[0])**2 + (yy - sensor[1])**2)
        covered = np.sum(distances <= coverage_radius)
        uncovered_area += region_size**2 - covered
        
        # Calculando conectividad (penalización si no está conectado)
        if distance_to_base > coverage_radius:
            disconnected_penalty += 1
            
    # Fitness = Minimizar energía + Minimizar área no cubierta + Minimizar desconexiones
    return energy_consumption + uncovered_area + disconnected_penalty * 1000

# Gravedad y movimiento
def gravitational_force(agent_i, agent_j, mass_j):
    distance = np.linalg.norm(agent_i - agent_j, axis=1)
    force = mass_j / (distance + 1e-5)[:, np.newaxis]  # Ajuste de dimensiones para evitar errores de índice
    return force

def update_position(agent, force, velocity):
    velocity = 0.5 * velocity + force  # Actualizar la velocidad
    agent += velocity  # Actualizar posición
    agent = np.clip(agent, 0, region_size)  # Limitar posiciones a la región
    return agent, velocity

# Inicialización de velocidades
velocities = np.zeros_like(agents)

# Ejecución del algoritmo GSA
best_fitness = np.inf
best_agent = None
fitness_history = []
agent_positions_history = []

for iteration in range(n_iterations):
    fitness_values = np.array([fitness(agent) for agent in agents])
    
    # Normalización de las masas de los agentes
    masses = (fitness_values.max() - fitness_values) / (fitness_values.max() - fitness_values.min() + 1e-5)
    
    # Calcular el centroide de cada agente (posición promedio de sus sensores) para seguimiento
    centroids = np.mean(agents, axis=1)
    agent_positions_history.append(np.copy(centroids))
    
    # Encontrar el mejor agente
    min_idx = np.argmin(fitness_values)
    if fitness_values[min_idx] < best_fitness:
        best_fitness = fitness_values[min_idx]
        best_agent = np.copy(agents[min_idx])
    
    fitness_history.append(best_fitness)
    
    # Calcular las fuerzas gravitacionales y actualizar posiciones
    for i in range(n_agents):
        total_force = np.zeros((n_sensors, 2))
        for j in range(n_agents):
            if i != j:
                total_force += gravitational_force(agents[i], agents[j], masses[j])
        agents[i], velocities[i] = update_position(agents[i], total_force, velocities[i])

# Visualización final de la mejor configuración encontrada
plt.figure(figsize=(10, 10))
plt.scatter(best_agent[:, 0], best_agent[:, 1], c='blue', label='Sensores')
plt.scatter(base_station[0], base_station[1], c='red', label='Estación base')
plt.xlim(0, region_size)
plt.ylim(0, region_size)
plt.title("Mejor Configuración de Sensores Encontrada por GSA")
plt.legend()
plt.grid(True)
plt.show()

# Graficar la evolución del fitness a lo largo de las iteraciones
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
    plt.scatter(base_station[0], base_station[1], color='red', marker='x', s=100, label='Estación Base')
    plt.xlim(0, region_size)
    plt.ylim(0, region_size)
    plt.title(f"Posiciones de los Agentes - Iteración {iter_num}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()



