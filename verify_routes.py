# Verify all saved routes match their claimed costs
Q, demands, dist_matrix = 478, [0], []
with open('instance.txt', 'r') as f:
    lines = f.readlines()
    Q = int(lines[0].strip())
    demands = [0] + [int(d) for d in lines[1].strip().split()]
    for i in range(2, len(lines)):
        dist_matrix.append([int(d) for d in lines[i].strip().split()])

saved_solutions = [
    {'iteration': 1, 'claimed_cost': 8742, 'routes': [[1, 11, 22, 7, 14, 25], [8, 10, 6, 17, 4, 23], [9], [13, 2, 24, 3, 5, 21], [20, 16, 12, 19, 18, 15]]},
    {'iteration': 2, 'claimed_cost': 8799, 'routes': [[1, 14, 7, 22, 11], [6, 4, 17, 20, 10, 25], [8, 9, 23], [15, 18, 19, 12, 16], [21, 5, 3, 24, 2, 13]]},
    {'iteration': 3, 'claimed_cost': 9099, 'routes': [[1, 14, 7, 22, 11], [6, 4, 17, 20, 10, 25], [9, 8, 23], [15, 18, 19, 12, 16], [21, 5, 3, 24, 2, 13]]},
    {'iteration': 4, 'claimed_cost': 9291, 'routes': [[1, 14, 7, 22, 11], [6, 4, 17, 20, 10, 25], [9, 8, 23], [15, 18, 13, 19, 12, 16], [21, 5, 3, 24, 2]]},
    {'iteration': 5, 'claimed_cost': 9291, 'routes': [[1, 14, 7, 22, 11], [6, 4, 17, 20, 10, 25], [9, 8, 23], [15, 18, 13, 19, 12, 16], [21, 5, 3, 24, 2]]},
]

print('Iteration | Claimed | Actual  | Match?')
print('-' * 45)
for sol in saved_solutions:
    total = 0
    for route in sol['routes']:
        d = dist_matrix[0][route[0]]
        for j in range(len(route)-1):
            d += dist_matrix[route[j]][route[j+1]]
        d += dist_matrix[route[-1]][0]
        total += d
    match = 'YES' if total == sol['claimed_cost'] else 'NO'
    print(f"{sol['iteration']:9} | {sol['claimed_cost']:7} | {total:7} | {match}")
