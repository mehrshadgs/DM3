import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import csv

# Use of AI
# Where is Commpleted by AI is indicated in comments,
# Ai helped with deubgging
# Print functions and Saving to CSV functions are AI generated mostly
# part d and e AI helped with idea and implementation
# Ai helped with reading the instance file


def read_cvrp_instance(filename):
    """
    Read CVRP instance from file.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Read vehicle capacity 
    Q = int(lines[0].strip())
    
    # Read customer demands
    demands_str = lines[1].strip().split()
    demands = [int(d) for d in demands_str]
    n = len(demands)  # number of customers
    
    # Insert depot demand (0) at the beginning
    demands = [0] + demands
    
    # Read distance matrix (remaining n+1 lines)
    distance_matrix = []
    for i in range(2, 2 + n + 1):
        row = [int(d) for d in lines[i].strip().split()]
        distance_matrix.append(row)
    
    return Q, demands, distance_matrix


def solve_cvrp_two_index(Q, demands, dist_matrix, time_limit=300):
    """
    Solves CVRP using the classical two-index formulation with MTZ subtour elimination.
    
    """
    n_nodes = len(demands)  # Total nodes (depot + customers)
    n_customers = n_nodes - 1  # Number of customers
    
    # Calculate minimum number of vehicles needed
    total_demand = sum(demands)
    min_vehicles = int(np.ceil(total_demand / Q))
    
    # print(f"Number of customers: {n_customers}")
    # print(f"Vehicle capacity: {Q}")
    # print(f"Total demand: {total_demand}")
    # print(f"Minimum vehicles needed: {min_vehicles}")
    
    model = gp.Model("CVRP_TwoIndex")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1)
    
    # Decision variables
    # x[i,j] = 1 if arc (i,j) is used
    x = model.addVars(n_nodes, n_nodes, vtype=GRB.BINARY)
    
    # u[i] = cumulative demand when arriving at customer i (MTZ variables for capacity)
    u = model.addVars(range(1, n_nodes), lb=0, ub=Q, vtype=GRB.CONTINUOUS)
    
    # Objective: minimize total distance
    model.setObjective(
        gp.quicksum(dist_matrix[i][j] * x[i, j] 
                    for i in range(n_nodes) for j in range(n_nodes) if i != j),
        GRB.MINIMIZE
    )
    
    # Each customer is visited exactly once (entering)
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in range(n_nodes) if i != j) == 1 
         for j in range(1, n_nodes)),
        name="enter"
    )
    
    # Each customer is left exactly once (leaving)
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in range(n_nodes) if i != j) == 1 
         for i in range(1, n_nodes))
    )
    
    # Number of vehicles leaving depot equals number returning
    model.addConstr(
        gp.quicksum(x[0, j] for j in range(1, n_nodes)) == 
        gp.quicksum(x[i, 0] for i in range(1, n_nodes))
    )
    
    # At least minimum number of vehicles must be used
    model.addConstr(
        gp.quicksum(x[0, j] for j in range(1, n_nodes)) >= min_vehicles
    )
    
    # this and next one with help of AI
    # u_i - u_j + Q*x_ij <= Q - q_j  for all i in N, j in N\{0}, i != j
    model.addConstrs(
        (u[i] - u[j] + Q * x[i, j] <= Q - demands[j]
         for i in range(1, n_nodes) for j in range(1, n_nodes) if i != j)
    )
    
    # u_0 - u_j + Q*x_0j <= Q - q_j
    # Since u_0 = 0 (9f): -u_j + Q*x_0j <= Q - q_j => u_j >= Q*x_0j - Q + q_j
    model.addConstrs(
        (u[j] >= demands[j] + Q * (x[0, j] - 1)
         for j in range(1, n_nodes))
    )
    
    #  q_i <= u_i <= Q for all i in N\{0}
    model.addConstrs(
        (u[i] >= demands[i] for i in range(1, n_nodes))
    )
    model.addConstrs(
        (u[i] <= Q for i in range(1, n_nodes))
    )
    
    print("\nSolving CVRP with two-index formulation...")
    print("-" * 50)
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    
    # Prepare results
    result = {
        "status": None,
        "n_customers": n_customers,
        "objective_value": None,
        "best_bound": None,
        "gap": None,
        "routes": [],
        "runtime": end_time - start_time,
        "num_vehicles": 0
    }
    
    # with help of AI
    if model.status == GRB.OPTIMAL:
        result["status"] = "Optimal"
        result["objective_value"] = model.objVal
        result["best_bound"] = model.objBound
        result["gap"] = 0.0
    elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        result["status"] = "Time Limit"
        result["objective_value"] = model.objVal
        result["best_bound"] = model.objBound
        result["gap"] = model.MIPGap * 100 
    elif model.status == GRB.INFEASIBLE:
        result["status"] = "Infeasible"
        return result
    else:
        result["status"] = f"Other ({model.status})"
        if model.SolCount > 0:
            result["objective_value"] = model.objVal
            result["best_bound"] = model.objBound
            result["gap"] = model.MIPGap * 100
        return result
    
    # Extract routes from solution 
    # Ai suggested to use another function
    if model.SolCount > 0:
        routes = extract_routes(x, n_nodes)
        result["routes"] = routes
        result["num_vehicles"] = len(routes)
    
    return result


def extract_routes(x, n_nodes): # this function is AI generated
    """
    Extract routes from the solution.
    """
    routes = []
    
    # Find all arcs leaving depot
    depot_arcs = [(0, j) for j in range(1, n_nodes) if x[0, j].X > 0.5]
    
    for _, first_customer in depot_arcs:
        route = [first_customer]
        current = first_customer
        
        # Follow the route until returning to depot
        while True:
            next_node = None
            for j in range(n_nodes):
                if j != current and x[current, j].X > 0.5:
                    next_node = j
                    break
            
            if next_node == 0 or next_node is None:
                break
            
            route.append(next_node)
            current = next_node
        
        routes.append(route)
    
    return routes


def print_solution(result, demands, dist_matrix, Q):
    """
    Print the solution details.
    """
    print("\n" + "=" * 60)
    print("CVRP SOLUTION REPORT (Two-Index Formulation)")
    print("=" * 60)
    
    print(f"\nStatus: {result['status']}")
    print(f"Computing Time: {result['runtime']:.2f} seconds")
    
    if result['objective_value'] is not None:
        print(f"\nTotal Cost (Distance): {result['objective_value']:.2f}")
        print(f"Best Bound: {result['best_bound']:.2f}")
        print(f"Optimality Gap: {result['gap']:.2f}%")
        print(f"Number of Vehicles Used: {result['num_vehicles']}")
        
        print("\n" + "-" * 40)
        print("ROUTES DETAIL:")
        print("-" * 40)
        
        total_distance_check = 0
        for idx, route in enumerate(result['routes'], 1):
            route_demand = sum(demands[c] for c in route)
            
            # Calculate route distance
            route_dist = dist_matrix[0][route[0]]  # Depot to first customer
            for i in range(len(route) - 1):
                route_dist += dist_matrix[route[i]][route[i+1]]
            route_dist += dist_matrix[route[-1]][0]  # Last customer to depot
            total_distance_check += route_dist
            
            print(f"\nRoute {idx}: 0 -> {' -> '.join(map(str, route))} -> 0")
            print(f"  Customers visited: {len(route)}")
            print(f"  Total demand: {route_demand} / {Q} (capacity)")
            print(f"  Route distance: {route_dist}")
        
        print("\n" + "-" * 40)
        print(f"Total distance (verification): {total_distance_check}")
    else:
        print("No feasible solution found.")


def simulate_demand_uncertainty(routes, nominal_demands, Q, n_iterations=1000, seed=None):
    """
    Simulate demand uncertainty to assess robustness of the solution.
    
    """
    # Track statistics per scenario
    scenario_violations = []  #
    
    
    for k in range(n_iterations):
        # Draw random demand for each customer
        # q_tilde[i] ~ Uniform[0.9*q[i], 1.1*q[i]] - continuous uniform
        random_demands = [0.0] * len(nominal_demands)  # depot demand = 0
        
        for i in range(1, len(nominal_demands)):
            q_i = nominal_demands[i]
            low = int(np.ceil(0.9 * q_i))
            high = int(np.floor(1.1 * q_i))
            random_demands[i] = np.random.randint(low, high + 1)
        
        # Compute capacity violations for each route and sum
        scenario_total_violation = 0
        for route in routes:
            route_demand = sum(random_demands[c] for c in route)
            violation = max(0, route_demand - Q)
            scenario_total_violation += violation
        
        scenario_violations.append(scenario_total_violation)
    
  
    scenario_violations = np.array(scenario_violations)
    
    scenarios_with_violation = np.sum(scenario_violations > 0)
    
    results = {
        "n_iterations": n_iterations,
        "n_routes": len(routes),
        "total_scenarios_with_violation": int(scenarios_with_violation),
        "mean_scenario_violation": np.mean(scenario_violations),
        "max_scenario_violation": np.max(scenario_violations),
        "min_scenario_violation": np.min(scenario_violations),
        "std_scenario_violation": np.std(scenario_violations),
        "percentage_scenarios_with_violation": (scenarios_with_violation / n_iterations) * 100,
        "scenario_violations": scenario_violations
    }
    
    return results


def print_simulation_results(sim_results): # this function is AI generated
    """
    Print the simulation results in a formatted way.
    """
    print("\n" + "=" * 60)
    print("PART 1.2(b): DEMAND UNCERTAINTY SIMULATION RESULTS")
    print("=" * 60)
    
    print(f"\nSimulation Parameters:")
    print(f"  Number of iterations (k): {sim_results['n_iterations']}")
    print(f"  Number of routes: {sim_results['n_routes']}")
    print(f"  Demand uncertainty: Uniform in [0.9·q_i, 1.1·q_i]")
    
    print("\n" + "-" * 40)
    print("CAPACITY VIOLATION STATISTICS")
    print("-" * 40)
    
    print(f"\n*** KEY RESULTS ***")
    print(f"  Total number of scenario violations: {sim_results['total_scenarios_with_violation']}")
    print(f"  Average (mean) capacity violation: {sim_results['mean_scenario_violation']:.4f}")
    print(f"  Maximum capacity violation: {sim_results['max_scenario_violation']:.4f}")
    
    print(f"\nAdditional Statistics:")
    print(f"  Scenarios with violation: {sim_results['total_scenarios_with_violation']} / {sim_results['n_iterations']}")
    print(f"    ({sim_results['percentage_scenarios_with_violation']:.2f}% of scenarios)")
    print(f"  Minimum violation: {sim_results['min_scenario_violation']:.4f}")
    print(f"  Std deviation: {sim_results['std_scenario_violation']:.4f}")
    
    # Distribution of violations
    violations = sim_results['scenario_violations']
    print(f"\nDistribution of scenario violations:")
    print(f"  Zero violations: {np.sum(violations == 0)} scenarios")
    print(f"  Violations (0, 10]: {np.sum((violations > 0) & (violations <= 10))} scenarios")
    print(f"  Violations (10, 25]: {np.sum((violations > 10) & (violations <= 25))} scenarios")
    print(f"  Violations > 25: {np.sum(violations > 25)} scenarios")


def save_simulation_to_csv(sim_results, filename): # this function is AI generated
    """
    Save simulation results to a CSV file.
    
    Args:
        sim_results: Dictionary containing simulation results
        filename: Name of the CSV file to save
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header and summary statistics
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Number of Iterations', sim_results['n_iterations']])
        writer.writerow(['Number of Routes', sim_results['n_routes']])
        writer.writerow(['Scenarios with Violation', sim_results['total_scenarios_with_violation']])
        writer.writerow(['Percentage with Violation', f"{sim_results['percentage_scenarios_with_violation']:.2f}%"])
        writer.writerow(['Mean Violation', f"{sim_results['mean_scenario_violation']:.4f}"])
        writer.writerow(['Max Violation', f"{sim_results['max_scenario_violation']:.4f}"])
        writer.writerow(['Min Violation', f"{sim_results['min_scenario_violation']:.4f}"])
        writer.writerow(['Std Deviation', f"{sim_results['std_scenario_violation']:.4f}"])
        writer.writerow([])
        
        # Per-scenario violations
        writer.writerow(['Scenario', 'Violation'])
        for i, violation in enumerate(sim_results['scenario_violations']):
            writer.writerow([i + 1, violation])
    
    print(f"\n  Part (b) results saved to '{filename}'")


def solve_cvrp_scenario_based(Q, nominal_demands, dist_matrix, scenarios, time_limit=600):
    """Solves the scenario-based CVRP. Routes must be feasible for ALL scenarios in S."""
    n_nodes = len(nominal_demands)
    n_customers = n_nodes - 1
    n_scenarios = len(scenarios)
    
   
    max_total_demand = max(sum(s) for s in scenarios)
    min_vehicles = int(np.ceil(max_total_demand / Q))
    
    # print(f"\n  Number of scenarios: {n_scenarios}")
    # print(f"  Minimum vehicles needed: {min_vehicles}")
    
    # Create model
    model = gp.Model("CVRP_ScenarioBased")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)  
    
    # Decision variables
    # x[i,j] = 1 if arc (i,j) is used (same for all scenarios - first-stage decision)
    x = model.addVars(n_nodes, n_nodes, vtype=GRB.BINARY)
    
    # u[s,i] = cumulative demand when arriving at customer i under scenario s
    u = {}
    for s in range(n_scenarios):
        for i in range(1, n_nodes):
            u[s, i] = model.addVar(lb=0, ub=Q, vtype=GRB.CONTINUOUS)
    
    # Objective: minimize total distance (same for all scenarios)
    model.setObjective(
        gp.quicksum(dist_matrix[i][j] * x[i, j] 
                    for i in range(n_nodes) for j in range(n_nodes) if i != j),
        GRB.MINIMIZE
    )
    
    #  Each customer is left exactly once
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in range(n_nodes) if i != j) == 1 
         for i in range(1, n_nodes))
    )
    
    #  Each customer is entered exactly once
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in range(n_nodes) if i != j) == 1 
         for j in range(1, n_nodes))
    )
    
    # Depot flow balance
    model.addConstr(
        gp.quicksum(x[0, j] for j in range(1, n_nodes)) == 
        gp.quicksum(x[i, 0] for i in range(1, n_nodes))
    )
    
    # Minimum vehicles
    model.addConstr(
        gp.quicksum(x[0, j] for j in range(1, n_nodes)) >= min_vehicles
    )
    
    # For EACH scenario s, add capacity constraints
    for s in range(n_scenarios):
        scenario_demands = scenarios[s]
        
        # Constraint (9d): MTZ subtour elimination with capacity for scenario s
        # u_i^s - u_j^s + Q*x_ij <= Q - q_j^s
        model.addConstrs(
            (u[s, i] - u[s, j] + Q * x[i, j] <= Q - scenario_demands[j]
             for i in range(1, n_nodes) for j in range(1, n_nodes) if i != j)
        )
        
        # Constraint (9d) for arcs from depot
        model.addConstrs(
            (u[s, j] >= scenario_demands[j] + Q * (x[0, j] - 1)
             for j in range(1, n_nodes))
        )
        
        # Constraint (9e): q_i^s <= u_i^s <= Q
        model.addConstrs(
            (u[s, i] >= scenario_demands[i] for i in range(1, n_nodes))
        )
        model.addConstrs(
            (u[s, i] <= Q for i in range(1, n_nodes))
        )
    
    # Optimize
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    
    # Prepare results
    result = {
        "status": None,
        "n_customers": n_customers,
        "n_scenarios": n_scenarios,
        "objective_value": None,
        "best_bound": None,
        "gap": None,
        "routes": [],
        "runtime": end_time - start_time,
        "num_vehicles": 0
    }
    
    if model.status == GRB.OPTIMAL:
        result["status"] = "Optimal"
        result["objective_value"] = model.objVal
        result["best_bound"] = model.objBound
        result["gap"] = 0.0
    elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        result["status"] = "Time Limit"
        result["objective_value"] = model.objVal
        result["best_bound"] = model.objBound
        result["gap"] = model.MIPGap * 100
    elif model.status == GRB.INFEASIBLE:
        result["status"] = "Infeasible"
        return result
    else:
        result["status"] = f"Other ({model.status})"
        if model.SolCount > 0:
            result["objective_value"] = model.objVal
            result["best_bound"] = model.objBound
            result["gap"] = model.MIPGap * 100
        return result
    
    # Extract routes
    if model.SolCount > 0:
        routes = extract_routes(x, n_nodes)
        result["routes"] = routes
        result["num_vehicles"] = len(routes)
    
    return result


def simulate_and_find_worst_scenario(routes, nominal_demands, Q, n_iterations=1000, seed=None):
    """
    Simulate demand uncertainty and find the worst-case scenario.
    
    Returns:
        tuple: (simulation_results, worst_scenario_demands, worst_violation)
    """
    if seed is not None:
        np.random.seed(seed)
    
    scenario_violations = []
    all_scenarios = []
    
    for k in range(n_iterations):
        # Draw random demands
        random_demands = [0.0] * len(nominal_demands)
        for i in range(1, len(nominal_demands)):
            q_i = nominal_demands[i]
            low = int(np.ceil(0.9 * q_i))
            high = int(np.floor(1.1 * q_i))
            random_demands[i] = np.random.randint(low, high + 1)
        
        # Compute total violation
        total_violation = 0
        for route in routes:
            route_demand = sum(random_demands[c] for c in route)
            violation = max(0, route_demand - Q)
            total_violation += violation
        
        scenario_violations.append(total_violation)
        all_scenarios.append(random_demands)
    
    scenario_violations = np.array(scenario_violations)
    
    # Find worst scenario
    worst_idx = np.argmax(scenario_violations)
    worst_violation = scenario_violations[worst_idx]
    worst_scenario = all_scenarios[worst_idx]
    
    # Compute stats
    scenarios_with_violation = np.sum(scenario_violations > 0)
    
    results = {
        "n_iterations": n_iterations,
        "total_scenarios_with_violation": int(scenarios_with_violation),
        "mean_scenario_violation": np.mean(scenario_violations),
        "max_scenario_violation": np.max(scenario_violations),
        "percentage_scenarios_with_violation": (scenarios_with_violation / n_iterations) * 100,
    }
    
    return results, worst_scenario, worst_violation


def run_cutting_plane_algorithm(Q, nominal_demands, dist_matrix, n_iterations=5, time_limit=600):
    """
    Run the cutting-plane algorithm for scenario-based CVRP.
    """

    n_nodes = len(nominal_demands)
    
    # Initialize scenario set S with only nominal demands
    scenarios = [nominal_demands.copy()]
    
    iteration_results = []
    
    for iteration in range(1, n_iterations + 1): # this is with help of AI
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")
        print(f"Current number of scenarios in S: {len(scenarios)}")
        
        # Step 1: Solve scenario-based model
        print(f"\nStep 1: Solving scenario-based CVRP...")
        result = solve_cvrp_scenario_based(Q, nominal_demands, dist_matrix, scenarios, time_limit)
        
        print(f"\n  Status: {result['status']}")
        print(f"  Computing time: {result['runtime']:.2f} seconds")
        print(f"  Routing costs: {result['objective_value']:.2f}" if result['objective_value'] else "  No solution")
        print(f"  Optimality gap: {result['gap']:.2f}%" if result['gap'] is not None else "  Gap: N/A")
        print(f"  Number of vehicles: {result['num_vehicles']}")
        
        if not result['routes']:
            print("  ERROR: No feasible solution found!")
            break
        
        # Print routes
        print(f"\n  Routes:")
        for idx, route in enumerate(result['routes'], 1):
            route_demand = sum(nominal_demands[c] for c in route)
            print(f"    Route {idx}: 0 -> {' -> '.join(map(str, route))} -> 0 (demand: {route_demand}/{Q})")
        
        # Step 2: Simulate 1000 demand realizations
        print(f"\nStep 2: Simulating 1000 demand realizations...")
        sim_results, worst_scenario, worst_violation = simulate_and_find_worst_scenario(
            routes=result['routes'],
            nominal_demands=nominal_demands,
            Q=Q,
            n_iterations=1000,
            seed=42 + iteration  # Different seed for each iteration
        )
        
        print(f"\n  Simulation Results:")
        print(f"    Total scenarios with violation: {sim_results['total_scenarios_with_violation']}")
        print(f"    Mean capacity violation: {sim_results['mean_scenario_violation']:.4f}")
        print(f"    Maximum capacity violation: {sim_results['max_scenario_violation']:.4f}")
        
        # Store iteration results
        iter_data = {
            "iteration": iteration,
            "n_scenarios": len(scenarios),
            "computing_time": result['runtime'],
            "optimality_gap": result['gap'],
            "routing_costs": result['objective_value'],
            "sim_violations": sim_results['total_scenarios_with_violation'],
            "sim_mean_violation": sim_results['mean_scenario_violation'],
            "sim_max_violation": sim_results['max_scenario_violation'],
            "worst_scenario_added": None,
            "routes": result['routes']
        }
        
        # Step 3: Add worst scenario if it has violations
        if worst_violation > 0:
            print(f"\nStep 3: Adding worst-case scenario to S")
            print(f"    Worst scenario violation: {worst_violation:.4f}")
            
            # Print customer demands of worst scenario
            print(f"    Customer demands of worst scenario:")
            demand_str = "    "
            for i in range(1, n_nodes):
                demand_str += f"q{i}={worst_scenario[i]:.1f}, "
                if i % 5 == 0:
                    print(demand_str)
                    demand_str = "    "
            if demand_str.strip():
                print(demand_str)
            
            scenarios.append(worst_scenario.copy())
            iter_data["worst_scenario_added"] = worst_scenario[1:]  # Exclude depot
        else:
            print(f"\nStep 3: No violations found - solution is robust!")
            print(f"    No scenario added to S")
        
        iteration_results.append(iter_data)
    

    
    # Save part (c) results to CSV
    save_cutting_plane_results_to_csv(iteration_results, "results_1_2c_cutting_plane.csv")
    
    return iteration_results

 
def save_cutting_plane_results_to_csv(results, filename): # this function is AI generated
    """Save cutting-plane algorithm results to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Num_Scenarios', 'Computing_Time_s', 'Optimality_Gap_pct',
                        'Routing_Costs', 'Num_Violations', 'Mean_Violation', 'Max_Violation'])
        
        for r in results:
            writer.writerow([
                r['iteration'],
                r['n_scenarios'],
                f"{r['computing_time']:.2f}",
                f"{r['optimality_gap']:.2f}",
                f"{r['routing_costs']:.2f}",
                r['sim_violations'],
                f"{r['sim_mean_violation']:.4f}",
                f"{r['sim_max_violation']:.4f}"
            ])
    
    print(f"\n  Part (c) results saved to '{filename}'")


def run_recourse_simulation_from_cutting_plane(cutting_plane_results, nominal_demands, dist_matrix, Q, n_sim_iters=1000):
    """
    Run recourse simulation for solutions from the cutting-plane algorithm.
    
    Args:
        cutting_plane_results: Results from run_cutting_plane_algorithm() containing routes for each iteration
        nominal_demands: Nominal demands
        dist_matrix: Distance matrix
        Q: Vehicle capacity
        n_sim_iters: Number of simulation iterations
    """
    print("\n" + "=" * 70)
    print("PART 1.2(e): RECOURSE POLICY SIMULATION")
    print("=" * 70)
    
    all_results = []
    
    for iter_data in cutting_plane_results:
        iteration = iter_data["iteration"]
        routes = iter_data["routes"]
        
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")
        print(f"Number of scenarios |S|: {iter_data['n_scenarios']}")
        print(f"Routing costs from optimization: {iter_data['routing_costs']:.2f}")
        
        # Print routes
        print(f"\nRoutes:")
        for idx, route in enumerate(routes, 1):
            route_demand = sum(nominal_demands[c] for c in route)
            print(f"  Route {idx}: 0 -> {' -> '.join(map(str, route))} -> 0 (demand: {route_demand}/{Q})")
        
        # Simulate recourse policy
        print(f"\nSimulating recourse policy (k={n_sim_iters} iterations)...")
        recourse_results = simulate_recourse_policy(
            routes=routes,
            nominal_demands=nominal_demands,
            dist_matrix=dist_matrix,
            Q=Q,
            n_iterations=n_sim_iters,
            seed=42 + iteration
        )
        
        print(f"\nRecourse Simulation Results:")
        print(f"  (i)  Average cost BEFORE recourse: {recourse_results['avg_cost_before_recourse']:.2f}")
        print(f"  (ii) Average cost AFTER recourse:  {recourse_results['avg_cost_after_recourse']:.2f}")
        avg_extra = recourse_results['avg_cost_after_recourse'] - recourse_results['avg_cost_before_recourse']
        print(f"  Average extra cost due to recourse: {avg_extra:.2f}")
        print(f"  Scenarios requiring recourse: {recourse_results['scenarios_with_recourse']}/{n_sim_iters} ({100*recourse_results['scenarios_with_recourse']/n_sim_iters:.1f}%)")
        print(f"  Average recourse actions per scenario: {recourse_results['avg_recourse_actions']:.2f}")
        
        result_data = {
            "iteration": iteration,
            "n_scenarios": iter_data['n_scenarios'],
            "routing_costs": iter_data['routing_costs'],
            "avg_cost_before_recourse": recourse_results['avg_cost_before_recourse'],
            "avg_cost_after_recourse": recourse_results['avg_cost_after_recourse'],
            "avg_extra_cost": avg_extra,
            "scenarios_with_recourse": recourse_results['scenarios_with_recourse'],
            "pct_with_recourse": 100 * recourse_results['scenarios_with_recourse'] / n_sim_iters,
            "avg_recourse_actions": recourse_results['avg_recourse_actions']
        }
        all_results.append(result_data)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("RECOURSE SIMULATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Iter':<6} {'|S|':<5} {'Avg Before':<16} {'Avg After':<16} {'Extra Cost':<14} {'%Recourse':<10}")
    print("-" * 75)
    
    for r in all_results:
        print(f"{r['iteration']:<6} {r['n_scenarios']:<5} "
              f"{r['avg_cost_before_recourse']:<16.2f} {r['avg_cost_after_recourse']:<16.2f} "
              f"{r['avg_extra_cost']:<14.2f} {r['pct_with_recourse']:<10.1f}%")
    
    # Save to CSV
    save_recourse_results_to_csv_v2(all_results, "results_1_2e_recourse.csv")
    
    
    return all_results


def save_recourse_results_to_csv_v2(results, filename): # this function is AI generated
    """Save recourse simulation results to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Num_Scenarios', 'Routing_Costs',
                        'Avg_Cost_Before_Recourse', 'Avg_Cost_After_Recourse',
                        'Avg_Extra_Cost', 'Scenarios_With_Recourse', 'Avg_Recourse_Actions'])
        
        for r in results:
            avg_extra = r['avg_cost_after_recourse'] - r['avg_cost_before_recourse']
            writer.writerow([
                r['iteration'],
                r['n_scenarios'],
                f"{r['routing_costs']:.2f}",
                f"{r['avg_cost_before_recourse']:.2f}",
                f"{r['avg_cost_after_recourse']:.2f}",
                f"{avg_extra:.2f}",
                r['scenarios_with_recourse'],
                f"{r['avg_recourse_actions']:.2f}"
            ])
    
    print(f"\n  Results saved to '{filename}'")



def simulate_recourse_policy(routes, nominal_demands, dist_matrix, Q, n_iterations=1000, seed=None):
    """
    Simulate the recourse policy for a given solution over multiple demand scenarios.
    """
    if seed is not None:
        np.random.seed(seed)
    
    costs_before_recourse = []
    costs_after_recourse = []
    recourse_actions_count = []
    
    for k in range(n_iterations):
        # Draw random demands
        random_demands = [0] * len(nominal_demands)
        for i in range(1, len(nominal_demands)):
            q_i = nominal_demands[i]
            low = int(np.ceil(0.9 * q_i))
            high = int(np.floor(1.1 * q_i))
            random_demands[i] = np.random.randint(low, high + 1)
        
        total_cost_before = 0
        total_cost_after = 0
        total_recourse_actions = 0
        
        for route in routes:
            # Cost before recourse (planned route cost)
            route_cost_planned = dist_matrix[0][route[0]]  # depot to first
            for j in range(len(route) - 1):
                route_cost_planned += dist_matrix[route[j]][route[j+1]]
            route_cost_planned += dist_matrix[route[-1]][0]  # last to depot
            total_cost_before += route_cost_planned
            
            # Execute route with recourse policy
            # Key: Vehicle must TRAVEL to customer first, THEN discover demand
            current_load = 0
            current_location = 0
            route_cost_actual = 0
            
            for customer in route:
                # Step 1: Travel to the customer (always happens first)
                route_cost_actual += dist_matrix[current_location][customer]
                current_location = customer
                
                # Step 2: Discover the true demand upon arrival
                customer_demand = random_demands[customer]
                
                # Step 3: Check if we can serve
                if current_load + customer_demand <= Q:
                    # Can serve - just add to load
                    current_load += customer_demand
                else:
                    # RECOURSE: Cannot serve - return to depot and come back
                    # Return to depot from current customer
                    route_cost_actual += dist_matrix[customer][0]
                    # Return to customer from depot
                    route_cost_actual += dist_matrix[0][customer]
                    # Now serve with fresh capacity
                    current_load = customer_demand
                    total_recourse_actions += 1
            
            # Return to depot at end
            route_cost_actual += dist_matrix[current_location][0]
            total_cost_after += route_cost_actual
        
        costs_before_recourse.append(total_cost_before)
        costs_after_recourse.append(total_cost_after)
        recourse_actions_count.append(total_recourse_actions)
    
    results = {
        "n_iterations": n_iterations,
        "avg_cost_before_recourse": np.mean(costs_before_recourse),
        "avg_cost_after_recourse": np.mean(costs_after_recourse),
        "avg_recourse_actions": np.mean(recourse_actions_count),
        "std_cost_after_recourse": np.std(costs_after_recourse),
        "max_cost_after_recourse": np.max(costs_after_recourse),
        "min_cost_after_recourse": np.min(costs_after_recourse),
        "scenarios_with_recourse": np.sum(np.array(recourse_actions_count) > 0),
        "costs_before": costs_before_recourse,
        "costs_after": costs_after_recourse,
        "recourse_counts": recourse_actions_count
    }
    
    return results


def illustrate_recourse_policy(nominal_demands, dist_matrix, Q):

    # Choose Route 4: [13, 2, 24, 3, 5, 21] with nominal demand 466
    route = [13, 2, 24, 3, 5, 21]
    
    # Create a demand scenario with ~10% increase (worst-case-ish)
    true_demands = nominal_demands.copy()


    nominal_route_demand = 0
    for c in route:
        print(f"  Customer {c}: q_{c} = {nominal_demands[c]}")
        nominal_route_demand += nominal_demands[c]
    print(f"  Total nominal demand: {nominal_route_demand}")
    
    # Create scenario with increased demands (simulating 10% uncertainty) # Ai helped here
    scenario_multipliers = {13: 1.10, 2: 1.08, 24: 1.10, 3: 1.09, 5: 1.10, 21: 1.08}
    
    for c in route:
        true_demands[c] = int(np.ceil(nominal_demands[c] * scenario_multipliers[c]))
    
    true_route_demand = 0
    for c in route:
        print(f"  Customer {c}: q̃_{c} = {true_demands[c]} (was {nominal_demands[c]})")
        true_route_demand += true_demands[c]

    
    # Simulate the route execution with recourse
    current_load = 0
    total_distance = 0
    current_location = 0  # Start at depot
    step = 1
    
    
    # Process each customer in route
    for customer in route:
        # Step 1: TRAVEL to customer first (always happens)
        distance_to_customer = dist_matrix[current_location][customer]
        total_distance += distance_to_customer
        current_location = customer
        

        # AI 

        customer_demand = true_demands[customer]
        print(f"         Demand revealed: q̃_{customer} = {customer_demand}")
        
        # Step 3: Check if we can serve
        if current_load + customer_demand <= Q:
            # Can serve normally
            current_load += customer_demand
            print(f"         Load after service: {current_load}/{Q}")
            print()
        else:
            # RECOURSE: Cannot serve - must return to depot and come back
            print(f"         RECOURSE TRIGGERED!")
            print(f"         Cannot serve: current load {current_load} + demand {customer_demand} = {current_load + customer_demand} > {Q}")
            
            # Return to depot from current customer
            distance_to_depot = dist_matrix[customer][0]
            total_distance += distance_to_depot
            print(f"         → Return to depot from customer {customer}, distance: {distance_to_depot}")
            
            # Return to customer from depot
            distance_back = dist_matrix[0][customer]
            total_distance += distance_back
            print(f"         → Return to customer {customer} from depot, distance: {distance_back}")
            print(f"         → Refill vehicle, now serve customer {customer}")
            
            # Now serve with fresh capacity
            current_load = customer_demand
            print(f"         Load after service: {current_load}/{Q}")
            print()
        
        step += 1
    
    # Return to depot at the end
    distance_to_depot = dist_matrix[current_location][0]
    total_distance += distance_to_depot
    print(f"Step {step}: Return to depot")
    print(f"         Distance: {distance_to_depot}")
    print(f"         Final load delivered: {current_load}")
    
    original_distance = dist_matrix[0][route[0]]
    for j in range(len(route) - 1):
        original_distance += dist_matrix[route[j]][route[j+1]]
    original_distance += dist_matrix[route[-1]][0]
    
 


if __name__ == "__main__":
    # Read the instance
    
    Q, demands, dist_matrix = read_cvrp_instance("instance.txt")
    
    # PART 1.2(a): Two-Index Formulation
    
    result_a = solve_cvrp_two_index(Q, demands, dist_matrix, time_limit=600)
    print_solution(result_a, demands, dist_matrix, Q)
  
    # PART 1.2(b): Demand Uncertainty Simulation

 
    sim_results_b = simulate_demand_uncertainty(
        routes=result_a['routes'],
        nominal_demands=demands,
        Q=Q,
        n_iterations=1000,
        seed=42
    )
    print_simulation_results(sim_results_b)
    save_simulation_to_csv(sim_results_b, "results_1_2b_simulation.csv")
    

    # PART 1.2(c): Scenario-Based CVRP with Cutting-Plane Algorithm
  
    
    cutting_plane_results = run_cutting_plane_algorithm(
        Q=Q,
        nominal_demands=demands,
        dist_matrix=dist_matrix,
        n_iterations=5,
        time_limit=600  
    )
    
  
    # PART 1.2(d): Recourse Policy Illustration 
    # Part d and e, AI helped with idea and how to implement
    illustrate_recourse_policy(demands, dist_matrix, Q)
    
   
    # PART 1.2(e): Recourse Simulation for All 5 Solutions

    recourse_results = run_recourse_simulation_from_cutting_plane(
        cutting_plane_results=cutting_plane_results,
        nominal_demands=demands,
        dist_matrix=dist_matrix,
        Q=Q,
        n_sim_iters=1000
    )
    
  
