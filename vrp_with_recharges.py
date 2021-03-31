from ortools.linear_solver import pywraplp
import numpy as np


def main():
    # Create the mip solver with the CBC backend.
    solver = pywraplp.Solver('SolveProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    data = {}
    coordinates = np.array(
        [[54.177909, 87.1071625], [54.154644, 87.1895599], [54.209695, 87.1238995], [54.1900136, 87.0737743],
         [54.1626852, 87.1229553], [54.1491651, 87.1049309], [54.1433334, 87.1317959], [54.1380541, 87.171278],
         [54.1501202, 87.2331619], [54.162233, 87.1895599], [54.1770048, 87.2292137], [54.1927756, 87.2426033],
         [54.1778085, 87.1809769], [54.1903652, 87.138319], [54.1921228, 87.1237278], [54.1789137, 87.0602989],
         [54.1825805, 87.02425],
         [54.2010102, 87.0358372], [54.2203852, 87.0302582], [54.2388483, 87.0353222], [54.2184783, 87.053175],
         [54.2306714, 87.0766926], [54.2441649, 87.0860481], [54.2286645, 87.101841], [54.236591, 87.1155739],
         [54.232327, 87.1590042], [54.2277112, 87.1196079], [54.2153165, 87.1081066], [54.2176753, 87.1706772],
         [54.2132587, 87.1471596], [54.177909, 87.1071625]])

    # calculating distance between two locations
    def spherical_dist(pos1, pos2, r=6371):
        pos1 = pos1 * np.pi / 180
        pos2 = pos2 * np.pi / 180
        cos_lat1 = np.cos(pos1[..., 0])
        cos_lat2 = np.cos(pos2[..., 0])
        cos_lat_d = np.cos(pos1[..., 0] - pos2[..., 0])
        cos_lon_d = np.cos(pos1[..., 1] - pos2[..., 1])
        return r * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

    # creating distances matrix
    distances_matrix = (np.array(spherical_dist(coordinates[:, None], coordinates)) * 1000).round(decimals=0)
    # choosing number of locations. n = number of locations + 1
    n = 31
    # first and last locations are the same depot
    for i in range(n):
        distances_matrix[n - 1, i] = distances_matrix[0, i]
        distances_matrix[i, n - 1] = distances_matrix[i, 0]
    # choosing battery capacity
    q = 33

    data['distance_matrix'] = distances_matrix[:n, :n].tolist()
    # choosing maximum number of vehicles
    data['num_vehicles'] = 3

    locations = {}
    for k in range(data['num_vehicles']):
        for i in range(n):
            for j in range(n):
                locations[i, j, k] = solver.IntVar(0, 1, 'x[%i,%i,%i]' % (i, j, k))

    # dict for order of visiting locations
    order = {}
    # list for charge at each location
    charge = [[[] for i in range(n)] for k in range(data['num_vehicles'])]
    for k in range(data['num_vehicles']):
        # 0,1,2,3 - depot and charging stations
        charge[k][0] = q
        charge[k][1] = q
        charge[k][2] = q
        charge[k][3] = q
        # first and last locations
        order[k, 0] = 0
        order[k, n - 1] = n - 1
        for i in range(1, n - 1):
            order[k, i] = solver.IntVar(1, n - 2, 'r[%i,%i]' % (k, i))
        for i in range(4, n):
            charge[k][i] = solver.IntVar(0, q, 'y[%i][%i]' % (k, i))

    # minimize distance traveled
    solver.Minimize(
        solver.Sum(
            [data['distance_matrix'][i][j] * locations[i, j, k] for k in range(data['num_vehicles']) for i in
             range(n - 1)
             for j in range(1, n)]))

    # Constraints
    # charge in the next location must be lower than in the previous location
    for k in range(data['num_vehicles']):
        for i in range(n - 1):
            for j in range(4, n):
                if i != j:
                    solver.Add(
                        charge[k][j] <= charge[k][i] - data['distance_matrix'][i][j] * locations[
                            i, j, k] * 0.001 + q * (
                                1 - locations[i, j, k]))

    # in any location charge must be enough to return to depot or visit charging station
    for k in range(data['num_vehicles']):
        for i in range(4, n - 1):
            solver.Add(charge[k][i] >= min(data['distance_matrix'][i][n - 1] * 0.001,
                                           data['distance_matrix'][i][1] * 0.001, data['distance_matrix'][i][2] * 0.001,
                                           data['distance_matrix'][i][3] * 0.001))

    # Miller-Tucker-Zemlin subtour elimination constraints
    for k in range(data['num_vehicles']):
        for i in range(n - 1):
            for j in range(1, n):
                solver.Add((order[k, i] - order[k, j] + n * locations[i, j, k]) <= n - 1)

    # one vehicle must enter location j one time
    for j in range(4, n - 1):
        solver.Add(solver.Sum(
            [locations[i, j, k] for k in range(data['num_vehicles']) for i in range(n - 1)]) == 1)

    # last location cant be visited more than one time
    for k in range(data['num_vehicles']):
        solver.Add(solver.Sum(
            [locations[i, n - 1, k] for i in range(1, n - 1)]) <= 1)

    # at least one vehicle must enter depot
    solver.Add(solver.Sum(
        [locations[i, n - 1, k] for k in range(data['num_vehicles']) for i in range(1, n - 1)]) >= 1)

    # one vehicle must leave location i one time
    for i in range(4, n - 1):
        solver.Add(solver.Sum(
            [locations[i, j, k] for k in range(data['num_vehicles']) for j in range(1, n)]) == 1)

    # at least one vehicle must leave depot
    solver.Add(solver.Sum(
        [locations[0, j, k] for k in range(data['num_vehicles']) for j in range(1, n - 1)]) >= 1)

    # every visited location except depot must be leaved
    for k in range(data['num_vehicles']):
        for h in range(1, n - 1):
            solver.Add((solver.Sum([(locations[i, h, k]) for i in range(n - 1)]) - solver.Sum(
                [(locations[h, j, k]) for j in range(1, n)])) == 0)

    solver.set_time_limit(300000)
    solver.Solve()

    # temp_order, temp_charge, sorted_charge_for_k - temporary variables
    temp_order = {}
    temp_charge = {}
    sorted_order = []
    sorted_charge = []
    sorted_charge_for_k = []
    # placing route and charge data in lists for each vehicle
    for k in range(data['num_vehicles']):
        temp_order[0] = 0
        for j in range(1, n - 1):
            if sum(locations[i, j, k].solution_value() for i in range(n - 1)) > 0:
                temp_order[j] = order[k, j].solution_value()
        # j start from 4 because charge stations dont have solution value
        for j in range(4, n):
            if sum(locations[i, j, k].solution_value() for i in range(n - 1)) > 0:
                temp_charge[j] = charge[k][j].solution_value()
        temp_charge[0] = q
        temp_charge[1] = q
        temp_charge[2] = q
        temp_charge[3] = q
        sorted_order.append(sorted(temp_order.items(), key=lambda kv: kv[1]))
        # placing charge values according to order
        for loc in sorted(temp_order.items(), key=lambda kv: kv[1]):
            sorted_charge_for_k.append(temp_charge[loc[0]])
        try:
            sorted_charge_for_k.append(temp_charge[n-1])
        except:
            pass
        sorted_charge.append(sorted_charge_for_k.copy())
        sorted_charge_for_k.clear()
        temp_order.clear()
        temp_charge.clear()


    # print results
    for k in range(data['num_vehicles']):
        print('Car %d route' % (k))
        # print(r[k, 0])
        print('%d -> ' % (order[k, 0]), end='')
        for j in range(1, len(sorted_order[k])):
            print('%d -> ' % sorted_order[k][j][0], end='')
        print('%d' % (order[k, n - 1]), end='\n')
        #     for i in range(len(sorted_charge[k])):
        #         print('%d -> ' % sorted_charge[k][i], end='')
        print('\n')

    print('Total cost = ', solver.Objective().Value())
    print()

    for k in range(data['num_vehicles']):
        for i in range(n):
            for j in range(n):

                if locations[i, j, k].solution_value() > 0:
                    print('Car %d From  %d to %d.  Cost = %d' % (
                        k,
                        i,
                        j,
                        data['distance_matrix'][i][j]))

    print()
    print("Time = ", solver.WallTime(), " milliseconds")


if __name__ == '__main__':
    main()
