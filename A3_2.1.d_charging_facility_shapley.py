"""
Assignment 3 - Question 2.1: Charging Facility Location with Shapley Value
Course: Decision-making in Transport and Mobility (1CM110)
"""

import gurobipy as gp
from gurobipy import GRB
import time


def parse_network(filename):
    """Parse network file. Format: origin destination distance"""
    nodes = set()
    arcs = []
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            origin, destination, distance = int(parts[0]), int(parts[1]), int(parts[2])
            nodes.add(origin)
            nodes.add(destination)
            arcs.append((origin, destination, distance))
    
    return nodes, arcs


def parse_od_pairs(filename):
    """Parse OD pairs file. Format: volume origin destination (with header)"""
    od_pairs = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            od_pairs.append({
                'volume': float(parts[0]),
                'origin': int(parts[1]),
                'destination': int(parts[2])
            })
    
    return od_pairs


def build_arc_sets(arcs, nodes):
    """Build outgoing and incoming arc dictionaries for each node"""
    outgoing = {i: [] for i in nodes}
    incoming = {i: [] for i in nodes}
    
    for (i, j, dist) in arcs:
        outgoing[i].append(j)
        incoming[j].append(i)
    
    return outgoing, incoming


def solve_charging_facility_location(nodes, arcs, od_pairs, Q=10):
    """
    Solve Capacitated Charging Facility Location Problem.
    
    min  sum y_i                                                    (16a)
    s.t. flow conservation                                          (16b)
         x_ij^k <= y_j  (if j != destination)                       (16c)
         sum of inflow <= Q * y_i                                   (16f)
         x >= 0, y in {0,1}                                         (16d,16e)
    """
    outgoing, incoming = build_arc_sets(arcs, nodes)
    K = range(len(od_pairs))
    N = nodes
    
    model = gp.Model("CFL")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 600)
    
    # Variables
    y = model.addVars(N, vtype=GRB.BINARY, name="y")
    x = {}
    for (i, j, dist) in arcs:
        for k in K:
            x[i, j, k] = model.addVar(lb=0, name=f"x_{i}_{j}_{k}")
    
    # Objective (16a): minimize facilities
    model.setObjective(gp.quicksum(y[i] for i in N), GRB.MINIMIZE)
    
    # Constraint (16b): Flow conservation
    for i in N:
        for k in K:
            if i == od_pairs[k]['origin']:
                d_ik = od_pairs[k]['volume']
            elif i == od_pairs[k]['destination']:
                d_ik = -od_pairs[k]['volume']
            else:
                d_ik = 0
            
            outflow = gp.quicksum(x[i, j, k] for j in outgoing[i] if (i, j, k) in x)
            inflow = gp.quicksum(x[j, i, k] for j in incoming[i] if (j, i, k) in x)
            model.addConstr(outflow - inflow == d_ik, name=f"flow_{i}_{k}")
    
    # Constraint (16c): Need facility at intermediate nodes
    for (i, j, dist) in arcs:
        for k in K:
            if j != od_pairs[k]['destination']:
                model.addConstr(x[i, j, k] <= y[j], name=f"facility_{i}_{j}_{k}")
    
    # Constraint (16f): Capacity
    for i in N:
        total_inflow = gp.quicksum(
            x[j, i, k] for j in incoming[i] for k in K if (j, i, k) in x
        )
        model.addConstr(total_inflow <= Q * y[i], name=f"capacity_{i}")
    
    # Solve
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    
    # Results
    results = {'status': model.status, 'solve_time': solve_time}
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        results['num_facilities'] = model.objVal
        results['facility_locations'] = [i for i in N if y[i].x > 0.5]
        results['gap'] = model.MIPGap * 100
    else:
        results['num_facilities'] = None
        results['facility_locations'] = None
        results['gap'] = None
    
    return results


def solve_for_coalition(nodes, arcs, od_pairs_dict, companies, Q=10):
    """Solve for a coalition of companies by combining their OD pairs"""
    combined_od_pairs = []
    for company in companies:
        combined_od_pairs.extend(od_pairs_dict[company])
    
    results = solve_charging_facility_location(nodes, arcs, combined_od_pairs, Q)
    return results['num_facilities'], results['facility_locations']


def calculate_shapley_value(coalition_costs):
    """
    Calculate Shapley value for 3-player game (A, B, C).
    
    Weights for n=3:
    - |S|=0: 2/6
    - |S|=1: 1/6  
    - |S|=2: 2/6
    """
    v = {
        frozenset(): 0,
        frozenset(['A']): coalition_costs[('A',)],
        frozenset(['B']): coalition_costs[('B',)],
        frozenset(['C']): coalition_costs[('C',)],
        frozenset(['A', 'B']): coalition_costs[('A', 'B')],
        frozenset(['A', 'C']): coalition_costs[('A', 'C')],
        frozenset(['B', 'C']): coalition_costs[('B', 'C')],
        frozenset(['A', 'B', 'C']): coalition_costs[('A', 'B', 'C')]
    }
    
    # Shapley for A: S = ∅, {B}, {C}, {B,C}
    phi_A = (2/6) * (v[frozenset(['A'])] - v[frozenset()]) + \
            (1/6) * (v[frozenset(['A', 'B'])] - v[frozenset(['B'])]) + \
            (1/6) * (v[frozenset(['A', 'C'])] - v[frozenset(['C'])]) + \
            (2/6) * (v[frozenset(['A', 'B', 'C'])] - v[frozenset(['B', 'C'])])
    
    # Shapley for B: S = ∅, {A}, {C}, {A,C}
    phi_B = (2/6) * (v[frozenset(['B'])] - v[frozenset()]) + \
            (1/6) * (v[frozenset(['A', 'B'])] - v[frozenset(['A'])]) + \
            (1/6) * (v[frozenset(['B', 'C'])] - v[frozenset(['C'])]) + \
            (2/6) * (v[frozenset(['A', 'B', 'C'])] - v[frozenset(['A', 'C'])])
    
    # Shapley for C: S = ∅, {A}, {B}, {A,B}
    phi_C = (2/6) * (v[frozenset(['C'])] - v[frozenset()]) + \
            (1/6) * (v[frozenset(['A', 'C'])] - v[frozenset(['A'])]) + \
            (1/6) * (v[frozenset(['B', 'C'])] - v[frozenset(['B'])]) + \
            (2/6) * (v[frozenset(['A', 'B', 'C'])] - v[frozenset(['A', 'B'])])
    
    return {'A': phi_A, 'B': phi_B, 'C': phi_C}


def main():
    # Configuration
    network_file = "network.txt"
    pairs_files = {'A': "pairsA.txt", 'B': "pairsB.txt", 'C': "pairsC.txt"}
    Q = 10
    
    # Load data
    print("Loading network...")
    nodes, arcs = parse_network(network_file)
    print(f"  Nodes: {len(nodes)}, Arcs: {len(arcs)}")
    
    od_pairs_dict = {}
    for company, pairs_file in pairs_files.items():
        od_pairs_dict[company] = parse_od_pairs(pairs_file)
    
    # =========================================================================
    # Question 2.1(c): Solve for each company
    # =========================================================================
    print("\n" + "=" * 60)
    print("Question 2.1(c): Solve for each company separately")
    print("=" * 60)
    
    for company in ['A', 'B', 'C']:
        results = solve_charging_facility_location(nodes, arcs, od_pairs_dict[company], Q)
        print(f"\nCompany {company}: {int(results['num_facilities'])} facilities")
        print(f"  Locations: {sorted(results['facility_locations'])}")
        print(f"  Time: {results['solve_time']:.2f}s")
    
    # =========================================================================
    # Question 2.1(d): Shapley Value
    # =========================================================================
    print("\n" + "=" * 60)
    print("Question 2.1(d): Shapley Value for Cost Allocation")
    print("=" * 60)
    
    # Solve all coalitions
    coalitions = [('A',), ('B',), ('C',), ('A','B'), ('A','C'), ('B','C'), ('A','B','C')]
    coalition_costs = {}
    coalition_locations = {}
    
    print("\nSolving all coalitions...")
    for coalition in coalitions:
        num, locs = solve_for_coalition(nodes, arcs, od_pairs_dict, coalition, Q)
        coalition_costs[coalition] = num
        coalition_locations[coalition] = locs
        print(f"  {str(coalition):<20} -> {int(num)} facilities")
    
    # Coalition costs table
    print("\n" + "-" * 60)
    print("Coalition Costs:")
    print("-" * 60)
    for coalition in coalitions:
        name = '{' + ', '.join(coalition) + '}'
        print(f"  {name:<15} {int(coalition_costs[coalition]):>3} facilities  {sorted(coalition_locations[coalition])}")
    
    # Shapley values
    shapley = calculate_shapley_value(coalition_costs)
    
    print("\n" + "-" * 60)
    print("Shapley Value Cost Allocation:")
    print("-" * 60)
    print(f"{'Company':<10} {'Standalone':<12} {'Shapley':<12} {'Savings':<12}")
    print("-" * 60)
    
    for company in ['A', 'B', 'C']:
        standalone = int(coalition_costs[(company,)])
        shapley_val = shapley[company]
        savings = standalone - shapley_val
        print(f"{company:<10} {standalone:<12} {shapley_val:<12.2f} {savings:<12.2f}")
    
    # Summary
    total_standalone = sum(coalition_costs[(c,)] for c in ['A', 'B', 'C'])
    total_shapley = sum(shapley.values())
    total_coalition = coalition_costs[('A', 'B', 'C')]
    
    print("-" * 60)
    print(f"{'Total':<10} {int(total_standalone):<12} {total_shapley:<12.2f}")
    print(f"\nGrand coalition cost: {int(total_coalition)}")
    print(f"Efficiency check: Shapley sum = {total_shapley:.2f} (should equal {int(total_coalition)})")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  Without cooperation: {int(total_standalone)} facilities")
    print(f"  With cooperation:    {int(total_coalition)} facilities")
    print(f"  Total savings:       {int(total_standalone - total_coalition)} facilities")


if __name__ == "__main__":
    main()