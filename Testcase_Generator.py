import json
import random

def generate_evrp_instance(
    num_nodes=10,
    num_customers=5,
    num_chargers=2,
    max_neighbors=5,
    max_distance=20,
    max_pickup=15,
    max_delivery=15,
    capacity=100,
    energy_cap=1000,
    alpha=0.01,
    charging_ports=2,
    vehicles=3
):
    nodes = list(range(num_nodes))
    random.shuffle(nodes)

    depot = nodes[0]
    customer_nodes = nodes[1:num_customers+1]
    charger_nodes = nodes[num_customers+1:num_customers+1+num_chargers]
    
    distances = {str(n): {} for n in nodes}
    customer_data = {}

    for node in nodes:
        possible_neighbors = [n for n in nodes if n != node]
        random.shuffle(possible_neighbors)
        num_edges = random.randint(3, max_neighbors)

        for neighbor in possible_neighbors[:num_edges]:
            dist = random.randint(5, max_distance)
            distances[str(node)][str(neighbor)] = dist
            distances[str(neighbor)][str(node)] = dist  # Ensure symmetry

    for c in customer_nodes:
        customer_data[str(c)] = {
            "pickup": random.randint(1, max_pickup),
            "delivery": random.randint(1, max_delivery)
        }

    evrp_data = {
        "nodes": nodes,
        "depot": depot,
        "customers": customer_data,
        "chargers": charger_nodes,
        "vehicles": vehicles,
        "capacity": capacity,
        "energy_cap": energy_cap,
        "alpha": alpha,
        "charging_ports": charging_ports,
        "distances": distances
    }

    return evrp_data

def generate_multiple_tests(num_tests=10):
    for i in range(1, num_tests + 1):
        instance = generate_evrp_instance()
        filename = f"input_{i}.json"
        with open(filename, "w") as f:
            json.dump(instance, f, indent=2)
        print(f"Generated: {filename}")

# Run generator
if __name__ == "__main__":
    generate_multiple_tests(num_tests=10)
import json
import random

def generate_evrp_instance(
    num_nodes=None,
    num_customers=5,
    num_chargers=2,
    max_neighbors=5,
    max_distance=20,
    max_pickup=15,
    max_delivery=15,
    capacity=100,
    energy_cap=1000,
    alpha=0.01,
    charging_ports=2,
    vehicles=None
):
    # Ensure minimum number of nodes = depot (1) + num_customers + num_chargers.
    min_nodes = num_customers + num_chargers + 1
    if num_nodes is None:
        num_nodes = random.randint(5, 15)
        num_nodes = max(num_nodes, min_nodes)
    else:
        if num_nodes < min_nodes:
            num_nodes = min_nodes

    # Ensure vehicles is randomly chosen if not provided: between 2 and 10.
    if vehicles is None:
        vehicles = random.randint(2, 10)

    # Generate list of node IDs and shuffle
    nodes = list(range(num_nodes))
    random.shuffle(nodes)

    # Define special nodes
    depot = nodes[0]
    customer_nodes = nodes[1:1 + num_customers]
    charger_nodes = nodes[1 + num_customers:1 + num_customers + num_chargers]

    
    distances = {str(n): {} for n in nodes}
    customer_data = {}

    for node in nodes:
        possible_neighbors = [n for n in nodes if n != node]
        random.shuffle(possible_neighbors)
        num_edges = random.randint(3, max_neighbors)

        for neighbor in possible_neighbors[:num_edges]:
            dist = random.randint(5, max_distance)
            distances[str(node)][str(neighbor)] = dist
            distances[str(neighbor)][str(node)] = dist 
            
    # Generate customer pickup and delivery data
    customer_data = {str(node): {
        "pickup": random.randint(1, max_pickup),
        "delivery": random.randint(1, max_delivery)
    } for node in customer_nodes}

    evrp_data = {
        "nodes": nodes,
        "depot": depot,
        "customers": customer_data,
        "chargers": charger_nodes,
        "vehicles": vehicles,
        "capacity": capacity,
        "energy_cap": energy_cap,
        "alpha": alpha,
        "charging_ports": charging_ports,
        "distances": distances
    }

    return evrp_data

def generate_multiple_tests(num_tests=10):
    filenames = []
    for i in range(1, num_tests + 1):
        data = generate_evrp_instance()
        filename = f"input_{i}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Generated: {filename}")
        filenames.append(filename)
    return filenames

if __name__ == "__main__":
    generate_multiple_tests(num_tests=10)
