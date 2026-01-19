"""
Assignment 3 - Question 2.1(c) & 2.1(d): Charging Facility Location
Course: Decision-making in Transport and Mobility (1CM110)
"""

import gurobipy as gp
from gurobipy import GRB
import time
from itertools import combinations


def parse_network(filename):
    """Parse the implicit network file."""
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
    """Parse the OD pairs file for a company."""
    od_pairs = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
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
    """Build dictionaries for efficient arc lookup."""
    outgoing = {i: [] for i in nodes}       # outgoing[i] = nodes reachable from i
    incoming = {i: [] for i in nodes}       # incoming[i] = nodes that can reach i
    arc_set = set()
    
    for (i, j, dist) in arcs:
        outgoing[i].append(j)
        incoming[j].append(i)
        arc_set.add((i, j))
    
    return outgoing, incoming, arc_set


def solve_charging_facility_location(nodes, arcs, od_pairs, Q=10, verbose=False):
    """Solve the Capacitated Charging Facility Location Problem."""
    outgoing, incoming, arc_set = build_arc_sets(arcs, nodes)
    K = range(len(od_pairs))
    N = nodes
    
    model = gp.Model("Charging_Facility_Location")
    model.setParam('OutputFlag', 1 if verbose else 0)
    model.setParam('TimeLimit', 600)
    
    y = model.addVars(N, vtype=GRB.BINARY, name="y")        # y[i]=1 if facility at node i
    
    x = {}
    for (i, j, dist) in arcs:
        for k in K:
            x[i, j, k] = model.addVar(lb=0, name=f"x_{i}_{j}_{k}")
    
    model.setObjective(gp.quicksum(y[i] for i in N), GRB.MINIMIZE)      # min total facilities
    
    # Flow conservation：outflow - inflow = demand
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
    
    # Flow requires facility at intermediate nodes
    '''
    with the help of Claude to exclude destination nodes following the charging logic
    '''
    for (i, j, dist) in arcs:
        for k in K:
            if j != od_pairs[k]['destination']:
                model.addConstr(x[i, j, k] <= y[j], name=f"facility_{i}_{j}_{k}")
    
    # Capacity constraint (exclude flow arriving at destination)
    '''
    Only counting flows that require recharging.
    Exclude the flow arriving at the final destination to avoid overcounting capacity.
    With the help of Claude to improve codes and 
    '''
    for i in N:
        total_inflow = gp.quicksum(
            x[j, i, k] 
            for j in incoming[i] 
            for k in K 
            if (j, i, k) in x and i != od_pairs[k]['destination']
        )
        model.addConstr(total_inflow <= Q * y[i], name=f"capacity_{i}")
    
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    
    results = {'status': model.status, 'solve_time': solve_time}
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        results['num_facilities'] = int(model.objVal)
        results['facility_locations'] = sorted([i for i in N if y[i].x > 0.5])
        results['gap'] = model.MIPGap * 100
    else:
        results['num_facilities'] = None
        results['facility_locations'] = None
        results['gap'] = None
    
    return results


def main():
    network_file = "network.txt"
    pairs_files = {'A': "pairsA.txt", 'B': "pairsB.txt", 'C': "pairsC.txt"}
    Q = 10
    companies = ['A', 'B', 'C']
    
    print("Loading network...")
    nodes, arcs = parse_network(network_file)
    print(f"  Nodes: {len(nodes)}, Arcs: {len(arcs)}")
    
    all_od_pairs = {}
    for company, pairs_file in pairs_files.items():
        all_od_pairs[company] = parse_od_pairs(pairs_file)
        print(f"  Company {company} OD pairs: {len(all_od_pairs[company])}")
    
    # Solve for all coalitions (for Shapley value)
    all_results = {}        
    
    print()
    print("=" * 70)
    print("Solving for all coalitions")
    print("=" * 70)
    
    for r in range(1, len(companies) + 1):      # r = coalition size (1, 2, 3)
        for combo in combinations(companies, r):
            combo_name = '+'.join(combo)
            
            combined_od_pairs = []      # merge OD pairs from all companies in coalition
            for company in combo:
                combined_od_pairs.extend(all_od_pairs[company])
            
            print(f"\n--- Coalition {combo_name} ({len(combined_od_pairs)} OD pairs) ---")
            results = solve_charging_facility_location(nodes, arcs, combined_od_pairs, Q)
            all_results[combo_name] = results
            
            if results['num_facilities'] is not None:
                print(f"  Facilities: {results['num_facilities']}, Time: {results['solve_time']:.2f}s")
                print(f"  Locations: {results['facility_locations']}")
            else:
                print(f"  No feasible solution!")
    
    # Summary table
    print()
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Coalition':<12} {'# Facilities':<15} {'Time (s)':<12} {'Locations'}")
    print("-" * 70)
    
    for r in range(1, len(companies) + 1):
        for combo in combinations(companies, r):
            combo_name = '+'.join(combo)
            res = all_results[combo_name]
            if res['num_facilities'] is not None:
                print(f"{combo_name:<12} {res['num_facilities']:<15} {res['solve_time']:<12.2f} {res['facility_locations']}")
            else:
                print(f"{combo_name:<12} {'N/A':<15} {res['solve_time']:<12.2f} N/A")
    
    # Shapley value calculation
    def v(coalition):       # cost function: facilities needed for coalition
        if len(coalition) == 0:
            return 0
        key = '+'.join(sorted(coalition))
        return all_results.get(key, {}).get('num_facilities', 0) or 0
    
    v_A, v_B, v_C = v({'A'}), v({'B'}), v({'C'})                        # standalone costs
    v_AB, v_AC, v_BC = v({'A', 'B'}), v({'A', 'C'}), v({'B', 'C'})      # pairwise coalitions
    v_ABC = v({'A', 'B', 'C'})                                          # grand coalition

    # the coefficients are taken from the lectures slide for three-party coalitions
    '''
    with the help of Claude to type the equations and the following "print" liens
    '''
    phi_A = (1/3)*v_A + (1/6)*(v_AB - v_B) + (1/6)*(v_AC - v_C) + (1/3)*(v_ABC - v_BC)
    phi_B = (1/3)*v_B + (1/6)*(v_AB - v_A) + (1/6)*(v_BC - v_C) + (1/3)*(v_ABC - v_AC)
    phi_C = (1/3)*v_C + (1/6)*(v_AC - v_A) + (1/6)*(v_BC - v_B) + (1/3)*(v_ABC - v_AB)
    
    print()
    print("=" * 70)
    print("SHAPLEY VALUE")
    print("=" * 70)
    print(f"{'Company':<12} {'Standalone':<12} {'Shapley':<12} {'Savings'}")
    print("-" * 70)
    print(f"{'A':<12} {v_A:<12} {phi_A:<12.4f} {v_A - phi_A:.4f}")
    print(f"{'B':<12} {v_B:<12} {phi_B:<12.4f} {v_B - phi_B:.4f}")
    print(f"{'C':<12} {v_C:<12} {phi_C:<12.4f} {v_C - phi_C:.4f}")
    print("-" * 70)
    print(f"{'Total':<12} {v_A+v_B+v_C:<12} {phi_A+phi_B+phi_C:<12.4f} (Grand coalition: {v_ABC})")


if __name__ == "__main__":
    main()