"""
Assignment 3 - Question 1.1(e): Last-Customer Formulation for Range Minimization
Course: Decision-making in Transport and Mobility (1CM110)
"""

import gurobipy as gp
from gurobipy import GRB
import time

# =============================================================================
# STEP 1: Parse the routes.txt file
# =============================================================================
def parse_routes(filename):
    """
    Parse the routes file and extract route information.
    """
    routes = []
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            
            distance = int(parts[0])
            load = int(parts[1])
            
            # Extract customer sequence (excluding depot 0 at start and end)
            # Format: 0 customer1 customer2 ... customerN 0
            full_sequence = [int(x) for x in parts[2:]]
            
            # Remove depot (0) from start and end to get customer list
            customers = [c for c in full_sequence if c != 0]
            
            if len(customers) > 0:
                last_customer = customers[-1]  # Last customer before returning to depot
            else:
                last_customer = None  # Empty route (shouldn't happen)
            
            routes.append({
                'distance': distance,           # route distance
                'load': load,                   # total demand on the route
                'customers': customers,         # list of customers visited (excluding depot)
                'last_customer': last_customer
            })
    
    return routes


# =============================================================================
# STEP 2: Build model parameters
# =============================================================================
def build_parameters(routes, num_customers):
    """
    Build the parameters needed for the Last-Customer Formulation.
    """
    R = range(len(routes))  # Route indices
    N = range(1, num_customers + 1)  # Customer indices (1 to n)
    
    p = {}  # Distance of each route
    c = {}  # Cost of each route
    a = {}  # a[i,r] = 1 if customer i is in route r
    b = {}  # b[i,r] = 1 if customer i is last customer in route r
    
    for r in R:
        p[r] = routes[r]['distance']
        c[r] = routes[r]['distance']  # Cost equals distance
        
        for i in N:
            # a[i,r] = 1 if customer i is visited by route r
            a[i, r] = 1 if i in routes[r]['customers'] else 0
            
            # b[i,r] = 1 if customer i is the last customer on route r
            b[i, r] = 1 if routes[r]['last_customer'] == i else 0
    
    return p, c, a, b


# =============================================================================
# STEP 3: z* from Question 1.1.a (already computed)
# =============================================================================
'''
The optimal cost z* = 8831.0 was obtained from solving the basic CVRP in Question 1.1(a). 
We use this value directly.
'''
Z_STAR = 8831.0


# =============================================================================
# STEP 4: Last-Customer Formulation for Range Minimization (Question 1.1.e)
# =============================================================================
def solve_last_customer_formulation(routes, num_customers, num_vehicles, z_star, epsilon):
    """
    Solve the Last-Customer Formulation for range minimization.
    
    Objective: min (eta - gamma)
    
    Subject to:
        (1b) Each customer visited exactly once
        (1c) Total cost <= (1 + epsilon) * z_star
        (1d) Select exactly |K| routes
        (1e) eta >= distance of each selected route (via last customer)
        (1f) gamma <= distance of each selected route (via last customer)

    With the help of Claude to form the equation of constraints and type the lines with the word "results"
    """
    R = range(len(routes))
    N = range(1, num_customers + 1)
    K = num_vehicles
    
    # Build parameters
    p, c, a, b = build_parameters(routes, num_customers)
    
    # Big-M: upper bound on route distance
    M = max(p[r] for r in R)
    
    # Cost cap
    cost_cap = (1 + epsilon) * z_star
    
    # Create model
    model = gp.Model("Last_Customer_Range_Minimization")
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output
    model.setParam('TimeLimit', 600)  # 10 minutes time limit
    
    # Decision variables
    x = model.addVars(R, vtype=GRB.BINARY, name="x")  # x[r] = 1 if route r is selected
    eta = model.addVar(lb=0, name="eta")  # Maximum route distance
    gamma = model.addVar(lb=0, name="gamma")  # Minimum route distance
    
    # Objective: minimize range (eta - gamma)
    model.setObjective(eta - gamma, GRB.MINIMIZE)
    
    # Constraint (1b): Each customer is visited exactly once
    for i in N:
        model.addConstr(
            gp.quicksum(a[i, r] * x[r] for r in R) == 1,
            name=f"visit_{i}"
        )
    
    # Constraint (1c): Total cost <= (1 + epsilon) * z_star
    model.addConstr(
        gp.quicksum(c[r] * x[r] for r in R) <= cost_cap,
        name="cost_bound"
    )
    
    # Constraint (1d): Select exactly K routes
    model.addConstr(
        gp.quicksum(x[r] for r in R) == K,
        name="num_vehicles"
    )
    
    # Constraint (1e): eta >= distance of each selected route (via last customer)
    # For each customer i: sum_r (p[r] * b[i,r] * x[r]) <= eta
    for i in N:
        model.addConstr(
            gp.quicksum(p[r] * b[i, r] * x[r] for r in R) <= eta,
            name=f"eta_bound_{i}"
        )
    
    # Constraint (1f): gamma <= distance of each selected route (via last customer)
    for i in N:
        model.addConstr(
            M * (1 - gp.quicksum(b[i, r] * x[r] for r in R)) +
            gp.quicksum(p[r] * b[i, r] * x[r] for r in R) >= gamma,
            name=f"gamma_bound_{i}"
        )
    
    # Solve
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    
    # Extract results
    results = {
        'epsilon': epsilon,
        'cost_cap': cost_cap,
        'status': model.status,
        'solve_time': solve_time
    }
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        selected_routes = [r for r in R if x[r].x > 0.5]
        selected_distances = [p[r] for r in selected_routes]
        
        results['total_cost'] = sum(c[r] for r in selected_routes)
        results['eta'] = eta.x
        results['gamma'] = gamma.x
        results['range'] = eta.x - gamma.x
        results['max_distance'] = max(selected_distances)
        results['min_distance'] = min(selected_distances)
        results['gap'] = model.MIPGap * 100
        results['selected_routes'] = selected_routes
    else:
        results['total_cost'] = None
        results['range'] = None
        results['gap'] = None
    
    return results


# =============================================================================
# STEP 5: Main function - Run for all epsilon values
# =============================================================================
'''
Output forming is with help of CLaude (these lines with the word "print")
'''
def main():
    # Configuration
    routes_file = "routes.txt"
    num_customers = 25
    num_vehicles = 5
    
    # Epsilon values to test
    epsilon_values = [0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15]
    
    # Parse routes
    print("Loading routes from file...")
    routes = parse_routes(routes_file)
    print(f"Loaded {len(routes)} routes")
    print()
    
    # Use z* from Question 1.1(a)
    z_star = Z_STAR
    print(f"Using z* = {z_star} from Question 1.1(a)")
    
    print()
    print("=" * 60)
    print("Solving Last-Customer Formulation for different epsilon values")
    print("(Question 1.1.e)")
    print("=" * 60)
    
    # Step 2: Solve Last-Customer Formulation for each epsilon
    all_results = []
    
    for eps in epsilon_values:
        print(f"\n--- Solving for epsilon = {eps} ---")
        results = solve_last_customer_formulation(
            routes, num_customers, num_vehicles, z_star, eps
        )
        all_results.append(results)
        
        if results['total_cost'] is not None:
            print(f"  Cost Cap: {results['cost_cap']:.1f}")
            print(f"  Total Cost: {results['total_cost']:.1f}")
            print(f"  Range (eta - gamma): {results['range']:.1f}")
            print(f"  Max Distance (eta): {results['eta']:.1f}")
            print(f"  Min Distance (gamma): {results['gamma']:.1f}")
            print(f"  Solving Time: {results['solve_time']:.2f} seconds")
            print(f"  Optimality Gap: {results['gap']:.2f}%")
        else:
            print(f"  No feasible solution found!")
    
    # Step 3: Print summary table
    print()
    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Epsilon':<10} {'Cost Cap':<12} {'Total Cost':<12} {'Range':<10} {'Max Dist':<10} {'Min Dist':<10} {'Time (s)':<10} {'Gap (%)':<10}")
    print("-" * 94)
    
    for res in all_results:
        if res['total_cost'] is not None:
            print(f"{res['epsilon']:<10.3f} {res['cost_cap']:<12.1f} {res['total_cost']:<12.1f} {res['range']:<10.1f} {res['max_distance']:<10.1f} {res['min_distance']:<10.1f} {res['solve_time']:<10.2f} {res['gap']:<10.2f}")
        else:
            print(f"{res['epsilon']:<10.3f} {res['cost_cap']:<12.1f} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {res['solve_time']:<10.2f} {'N/A':<10}")
    
    print()
    print("=" * 60)
    print("Analysis:")
    print("=" * 60)
    print("As epsilon increases:")
    print("  - Cost cap increases, allowing more flexibility")
    print("  - Range should decrease (better fairness)")
    print("  - Trade-off between efficiency and fairness")


if __name__ == "__main__":
    main()