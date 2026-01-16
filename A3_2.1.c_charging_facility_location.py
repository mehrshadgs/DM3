"""
Assignment 3 - Question 2.1: Charging Facility Location
Course: Decision-making in Transport and Mobility (1CM110)
"""

import gurobipy as gp
from gurobipy import GRB
import time


# =============================================================================
# STEP 1: Parse input files
# =============================================================================
def parse_network(filename):
    """
    Parse the implicit network file.
    
    Each line format: origin destination distance
    
    Returns:
        nodes: set of all node IDs
        arcs: list of tuples (i, j, distance)
    """
    nodes = set()
    arcs = []
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            
            origin = int(parts[0])
            destination = int(parts[1])
            distance = int(parts[2])
            
            nodes.add(origin)
            nodes.add(destination)
            arcs.append((origin, destination, distance))
    
    return nodes, arcs


def parse_od_pairs(filename):
    """
    Parse the OD pairs file for a company.
    
    Each line format: volume origin destination
    First line is header.
    
    Returns:
        od_pairs: list of dictionaries with keys 'volume', 'origin', 'destination'
    """
    od_pairs = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        # Skip header line
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            
            volume = float(parts[0])
            origin = int(parts[1])
            destination = int(parts[2])
            
            od_pairs.append({
                'volume': volume,
                'origin': origin,
                'destination': destination
            })
    
    return od_pairs


# =============================================================================
# STEP 2: Build helper data structures
# =============================================================================
def build_arc_sets(arcs, nodes):
    """
    Build dictionaries for efficient arc lookup.
    
    Returns:
        outgoing[i]: list of nodes j such that (i,j) is an arc
        incoming[i]: list of nodes j such that (j,i) is an arc
        arc_set: set of all arcs (i,j)
    """
    outgoing = {i: [] for i in nodes}
    incoming = {i: [] for i in nodes}
    arc_set = set()
    
    for (i, j, dist) in arcs:
        outgoing[i].append(j)
        incoming[i].append(j)  # j -> i means j is in incoming[i]
        arc_set.add((i, j))
    
    # Fix incoming: incoming[i] should contain j such that (j, i) exists
    incoming = {i: [] for i in nodes}
    for (i, j, dist) in arcs:
        incoming[j].append(i)
    
    return outgoing, incoming, arc_set


# =============================================================================
# STEP 3: Capacitated Charging Facility Location Model
# =============================================================================
def solve_charging_facility_location(nodes, arcs, od_pairs, Q=10, verbose=False):
    """
    Solve the Capacitated Charging Facility Location Problem.
    
    Model from lecture slides (16a-16e) with capacity constraint (16f):
    
    min  sum_{i in N} y_i                                           (16a)
    s.t. sum_{j:(i,j) in A} x_ij^k - sum_{j:(j,i) in A} x_ji^k = d_i^k   (16b)
         x_ij^k <= y_j                 for j != D_k                  (16c)
         sum_k sum_{j:(j,i) in A} x_ji^k <= Q * y_i                  (16f)
         x_ij^k >= 0                                                 (16d)
         y_i in {0, 1}                                               (16e)
    
    Parameters:
        nodes: set of node IDs
        arcs: list of (origin, destination, distance) tuples
        od_pairs: list of OD pair dictionaries
        Q: capacity of each charging facility
        verbose: whether to print Gurobi output
    
    Returns:
        Dictionary containing results
    """
    # Build arc sets
    outgoing, incoming, arc_set = build_arc_sets(arcs, nodes)
    
    # Index OD pairs
    K = range(len(od_pairs))
    N = nodes
    
    # Create model
    model = gp.Model("Charging_Facility_Location")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', 600)  # 10 minutes time limit
    
    # Decision variables
    # y[i] = 1 if facility is opened at location i
    y = model.addVars(N, vtype=GRB.BINARY, name="y")
    
    # x[i,j,k] = flow of OD pair k along arc (i,j)
    x = {}
    for (i, j, dist) in arcs:
        for k in K:
            x[i, j, k] = model.addVar(lb=0, name=f"x_{i}_{j}_{k}")
    
    # Objective (16a): minimize number of facilities
    model.setObjective(gp.quicksum(y[i] for i in N), GRB.MINIMIZE)
    
    # Constraint (16b): Flow conservation
    # sum_{j:(i,j) in A} x_ij^k - sum_{j:(j,i) in A} x_ji^k = d_i^k
    # where d_i^k = V_k if i = O_k, -V_k if i = D_k, 0 otherwise
    for i in N:
        for k in K:
            # Compute d_i^k
            if i == od_pairs[k]['origin']:
                d_ik = od_pairs[k]['volume']
            elif i == od_pairs[k]['destination']:
                d_ik = -od_pairs[k]['volume']
            else:
                d_ik = 0
            
            # Outgoing flow from i
            outflow = gp.quicksum(x[i, j, k] for j in outgoing[i] if (i, j, k) in x)
            
            # Incoming flow to i
            inflow = gp.quicksum(x[j, i, k] for j in incoming[i] if (j, i, k) in x)
            
            model.addConstr(outflow - inflow == d_ik, name=f"flow_{i}_{k}")
    
    # Constraint (16c): Flow requires facility at intermediate nodes
    # x_ij^k <= y_j for all k, (i,j) in A where j != D_k
    for (i, j, dist) in arcs:
        for k in K:
            if j != od_pairs[k]['destination']:
                model.addConstr(x[i, j, k] <= y[j], name=f"facility_{i}_{j}_{k}")
    
    # Constraint (16f): Capacity constraint
    # sum_k sum_{j:(j,i) in A} x_ji^k <= Q * y_i for all i in N
    for i in N:
        total_inflow = gp.quicksum(
            x[j, i, k] 
            for j in incoming[i] 
            for k in K 
            if (j, i, k) in x
        )
        model.addConstr(total_inflow <= Q * y[i], name=f"capacity_{i}")
    
    # Solve
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    
    # Extract results
    results = {
        'status': model.status,
        'solve_time': solve_time
    }
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        results['num_facilities'] = model.objVal
        results['facility_locations'] = [i for i in N if y[i].x > 0.5]
        results['gap'] = model.MIPGap * 100
    else:
        results['num_facilities'] = None
        results['facility_locations'] = None
        results['gap'] = None
    
    return results


# =============================================================================
# STEP 4: Main function - Solve for all three companies
# =============================================================================
def main():
    # Configuration
    network_file = "network.txt"
    pairs_files = {
        'A': "pairsA.txt",
        'B': "pairsB.txt",
        'C': "pairsC.txt"
    }
    Q = 10  # Capacity
    
    # Parse network
    print("Loading network...")
    nodes, arcs = parse_network(network_file)
    print(f"  Nodes: {len(nodes)}")
    print(f"  Arcs: {len(arcs)}")
    print()
    
    # Solve for each company
    print("=" * 60)
    print("Question 2.1(c): Solve for each company separately")
    print("=" * 60)
    
    all_results = {}
    
    for company, pairs_file in pairs_files.items():
        print(f"\n--- Company {company} ---")
        
        # Parse OD pairs
        od_pairs = parse_od_pairs(pairs_file)
        print(f"  OD pairs: {len(od_pairs)}")
        
        # Solve
        results = solve_charging_facility_location(nodes, arcs, od_pairs, Q)
        all_results[company] = results
        
        if results['num_facilities'] is not None:
            print(f"  Optimal number of facilities: {int(results['num_facilities'])}")
            print(f"  Facility locations: {sorted(results['facility_locations'])}")
            print(f"  Solving time: {results['solve_time']:.2f} seconds")
            print(f"  Optimality gap: {results['gap']:.2f}%")
        else:
            print(f"  No feasible solution found!")
    
    # Summary table
    print()
    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Company':<10} {'# Facilities':<15} {'Time (s)':<12} {'Locations'}")
    print("-" * 60)
    
    for company in ['A', 'B', 'C']:
        res = all_results[company]
        if res['num_facilities'] is not None:
            locs = sorted(res['facility_locations'])
            print(f"{company:<10} {int(res['num_facilities']):<15} {res['solve_time']:<12.2f} {locs}")
        else:
            print(f"{company:<10} {'N/A':<15} {res['solve_time']:<12.2f} N/A")


if __name__ == "__main__":
    main()