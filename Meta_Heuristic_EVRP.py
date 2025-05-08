import json
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from colorama import Fore, Style, init
import sys
import glob

init(autoreset=True)

# -------------------- DATA HANDLING -------------------- #
def load_data(filename):
    with open(filename) as f:
        data = json.load(f)
    
    data['nodes'] = [int(n) for n in data['nodes']]
    data['depot'] = int(data['depot'])
    data['chargers'] = [int(c) for c in data['chargers']]
    
    # Ensure all non-depot/charger nodes are customers
    all_nodes = set(data['nodes'])
    customers = {str(k): v for k, v in data.get('customers', {}).items()}
    missing_nodes = all_nodes - {data['depot']} - set(data['chargers']) - set(map(int, customers.keys()))
    
    for node in missing_nodes:
        customers[str(node)] = {'pickup': 0, 'delivery': 0}
    data['customers'] = customers
    
    # Convert distance matrix keys to integers
    data['distances'] = {int(k): {int(kk): int(vv) for kk, vv in v.items()} 
                        for k, v in data['distances'].items()}
    
    return data

# -------------------- GENETIC ALGORITHM CORE -------------------- #
def create_valid_individual(all_nodes, depot, vehicles, data, max_attempts=500):
    nodes = all_nodes.copy()
    random.shuffle(nodes)
    for _ in range(max_attempts):
        current_nodes = nodes.copy()
        temp_routes = []
        for v in range(vehicles):
            remaining_vehicles = vehicles - len(temp_routes)
            remaining_nodes = len(current_nodes)
            if remaining_nodes == 0:
                break
            min_nodes = max(1, remaining_nodes // remaining_vehicles)
            max_possible = min(remaining_nodes, min_nodes + 3)
            route_length = random.randint(
                min(min_nodes, max_possible),
                max(min_nodes, max_possible)
            ) if remaining_nodes >= min_nodes else remaining_nodes
            route_nodes = current_nodes[:route_length]
            del current_nodes[:route_length]
            route = [depot] + route_nodes + [depot]
            if validate_route(route, data, v):
                temp_routes.append(route)
            else:
                break
        if len(temp_routes) == vehicles and not current_nodes:
            return temp_routes
    # Fallback: balanced distribution
    chunk = max(1, len(all_nodes) // vehicles)
    return [[depot] + all_nodes[i*chunk:(i+1)*chunk] + [depot] 
            for i in range(vehicles)]

def validate_route(route, data, vehicle_idx):
    """Validate route constraints including load capacity"""
    current_load = data['initial_weights'][vehicle_idx]
    for node in route[1:-1]:  # Skip depot at start/end
        if str(node) in data['customers']:
            current_load += data['customers'][str(node)]['pickup']
            current_load -= data['customers'][str(node)]['delivery']
            if current_load < 0 or current_load > data['capacity']:
                return False
    return True

def calculate_fitness(individual, data, all_nodes):
    total_energy = 0
    visited = set()
    unique_routes = set()
    for idx, route in enumerate(individual):
        if len(route) < 3:  # Invalid empty route
            return float('inf')
        route_tuple = tuple(route)
        if route_tuple in unique_routes:
            return float('inf')
        unique_routes.add(route_tuple)
        energy = 0
        current_load = data['initial_weights'][idx]
        current_energy = data['energy_cap']
        for i in range(len(route)-1):
            from_node = route[i]
            to_node = route[i+1]
            try:
                distance = data['distances'][from_node][to_node]
            except KeyError:
                return float('inf')
            # Energy calculation before load changes
            energy_used = data['alpha'] * distance * current_load
            energy += energy_used
            current_energy -= energy_used
            # Update load after movement
            if str(to_node) in data['customers']:
                current_load += data['customers'][str(to_node)]['pickup']
                current_load -= data['customers'][str(to_node)]['delivery']
            # Smart charging strategy
            if current_energy < 150 and to_node in data['chargers']:
                current_energy = data['energy_cap']
                energy += data['alpha'] * distance * current_load  # Charging cost
        visited.update(route[1:-1])
        total_energy += energy
    missing = len(set(all_nodes) - visited)
    return total_energy + (1e6 * missing) + (len(individual) * 1e3)

def ordered_crossover(parent1, parent2, depot, vehicles):
    flat1 = [n for route in parent1 for n in route if n != depot]
    flat2 = [n for route in parent2 for n in route if n != depot]
    start, end = sorted(random.sample(range(len(flat1)), 2))
    child = flat1[start:end] + [n for n in flat2 if n not in flat1[start:end]]
    chunk_size = max(1, len(child) // vehicles)
    routes = []
    for i in range(vehicles):
        segment = child[i*chunk_size:(i+1)*chunk_size] if i < vehicles-1 else child[i*chunk_size:]
        if segment:
            routes.append([depot] + segment + [depot])
    return routes[:vehicles]

def mutate(individual, depot, mutation_rate=0.15):
    for route in individual:
        if len(route) > 3 and random.random() < mutation_rate:
            i, j = random.sample(range(1, len(route)-1), 2)
            route[i], route[j] = route[j], route[i]
    return individual

def genetic_algorithm(data):
    depot = data['depot']
    vehicles = data['vehicles']
    all_nodes = [n for n in data['nodes'] if n != depot]
    pop_size = max(100, len(all_nodes) * 2)
    max_gen = max(200, len(all_nodes) * 5)
    population = [create_valid_individual(all_nodes, depot, vehicles, data) 
                 for _ in range(pop_size)]
    for gen in range(max_gen):
        population.sort(key=lambda x: calculate_fitness(x, data, all_nodes))
        elites = population[:10]
        offspring = []
        while len(offspring) < pop_size - len(elites):
            parents = random.choices(population[:50], k=2)
            child = ordered_crossover(parents[0], parents[1], depot, vehicles)
            offspring.append(mutate(child, depot))
        population = elites + offspring
    best = min(population, key=lambda x: calculate_fitness(x, data, all_nodes))
    return [route for route in best if len(route) > 2]

# -------------------- SOLUTION VISUALIZATION -------------------- #
def print_solution_details(routes, data):
    print(Fore.CYAN + "\n" + "="*50)
    print(Fore.YELLOW + "⚡ OPTIMIZED ELECTRIC VEHICLE ROUTING SOLUTION")
    print(Fore.CYAN + "="*50)
    total_distance = 0
    total_energy = 0
    charging_stops = 0
    all_nodes = set(data['nodes'])
    visited = set()
    for i, route in enumerate(routes, 1):
        if len(route) < 3: continue
        route_distance = 0
        route_energy = 0
        current_load = data['initial_weights'][i-1]
        max_load = current_load
        current_energy = data['energy_cap']
        print(Fore.MAGENTA + f"\n🚚 VEHICLE {i} DETAILS:")
        print(Fore.WHITE + f"Route: {' → '.join(map(str, route))}")
        for j in range(len(route)-1):
            from_node = route[j]
            to_node = route[j+1]
            try:
                distance = data['distances'][from_node][to_node]
            except KeyError:
                continue
            energy_used = data['alpha'] * distance * current_load
            route_distance += distance
            route_energy += energy_used
            current_energy -= energy_used
            if str(to_node) in data['customers']:
                current_load += data['customers'][str(to_node)]['pickup']
                current_load -= data['customers'][str(to_node)]['delivery']
                max_load = max(max_load, current_load)
            if current_energy < 150 and to_node in data['chargers']:
                charging_stops += 1
                current_energy = data['energy_cap']
        visited.update(route[1:-1])
        total_distance += route_distance
        total_energy += route_energy
        print(Fore.CYAN + f"  📏 Distance: {route_distance} units")
        print(Fore.BLUE + f"  🔋 Energy: {route_energy:.2f} units")
        print(Fore.WHITE + f"  ⚖️ Max Load: {max_load}/{data['capacity']}")
        print(Fore.GREEN + f"  Initial Weight: {data['initial_weights'][i-1]} kg")
    missing = all_nodes - visited
    print(Fore.CYAN + "\n" + "="*50)
    print(Fore.YELLOW + "📊 SOLUTION SUMMARY")
    print(Fore.CYAN + "="*50)
    print(Fore.GREEN + f"Vehicles Used: {len(routes)}/{data['vehicles']}")
    print(Fore.BLUE + f"Total Distance: {total_distance} units")
    print(Fore.MAGENTA + f"Total Energy: {total_energy:.2f} units")

def visualize_solution(routes, data):
    G = nx.DiGraph()
    depot = data['depot']
    pos = nx.circular_layout(sorted(data['nodes']))
    pos[depot] = np.array([0, 0])
    for node in data['nodes']:
        if node == depot:
            G.add_node(node, color='red', size=300)
        elif node in data['chargers']:
            G.add_node(node, color='green', size=200)
        else:
            G.add_node(node, color='skyblue', size=150)
    colors = plt.cm.tab10.colors
    for i, route in enumerate(routes):
        route_color = colors[i % len(colors)]
        for j in range(len(route)-1):
            G.add_edge(route[j], route[j+1], color=route_color, width=2)
    plt.figure(figsize=(12, 8))
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    edge_colors = [G.edges[e]['color'] for e in G.edges()]
    edge_widths = [G.edges[e]['width'] for e in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, 
                          arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    plt.title("Optimized EV Routing Solution", pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('evrp_solution__1.png', dpi=300)
    plt.show()

# -------------------- MAIN EXECUTION -------------------- #
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        temp_file = input("Enter input filename: ")
        input_files = glob.glob(temp_file)
        filename = random.choice(input_files) if input_files else None
    if not filename:
        print(Fore.RED + "No input files found!")
        sys.exit(1)
    print(Fore.YELLOW + f"\n🚀 Processing: {filename}")
    data = load_data(filename)
    # Generate random initial weights for each vehicle (e.g., 25-50)
    data['initial_weights'] = [random.randint(25, 50) for _ in range(data['vehicles'])]
    best_routes = genetic_algorithm(data)
    print_solution_details(best_routes, data)
    visualize_solution(best_routes, data)
