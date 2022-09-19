import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import veroviz as vrv


# Source paper: 'Safe distance-based vehicle routing problem:
# Medical waste collection case study in COVID-19 pandemic'


# API Key for ORS geographical data to provide road network
# Website Link: https://openrouteservice.org/dev/#/home
ORS_API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'


'''
Request module SSL verification error handling
If you have any SSL error (ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] ) while running the code,
run the below commented code to solve the issue.
'''
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context


# Hospital data loading
# Data preprocessing after loading is necessary
H = np.genfromtxt('hospital_data.csv', dtype=None, delimiter=",", encoding='UTF-8')
coord_data = np.transpose(H)

# Making a list of integrated coordinates of every tertiary general hospital
hospital_coord_str = []
for num in range(1, coord_data.shape[1]):
    if coord_data[4][num] in ['G001', 'G006', 'G099']:
        hospital_coord_str.append([coord_data[-3][num], coord_data[-4][num]])

# Converting elements of hospital coordinates list from string to float data type
hospital_coord = []
for str_list in hospital_coord_str:
    float_list = list(map(float, str_list))
    hospital_coord.append(float_list)

# Making a list of hospital safety scores of every tertiary general hospital
hospital_safety_str = []
for num in range(1, coord_data.shape[1]):
    if coord_data[4][num] in ['G001', 'G006', 'G099']:
        hospital_safety_str.append(coord_data[-1][num])

# Converting elements of hospital safety scores list from string to float data type
hospital_safety = []
for string in hospital_safety_str:
    hospital_safety.append(int(string))

# Specifying list of hospitals
hospital_list = []
for i in range(len(hospital_coord)):
    hospital_list.append('H' + str(i+1))


# Create hospital nodes
myNodes = vrv.createNodesFromLocs(locs=hospital_coord, leafletIconPrefix='fa', leafletIconType='ambulance')

# Create time matrix and distance matrix in a dictionary form
[timeSec, distMeters] = vrv.getTimeDist2D(nodes        = myNodes,
                                          outputDistUnits  = 'km',
                                          routeType    = 'fastest',
                                          dataProvider = 'ORS-online',
                                          dataProviderArgs = { 'APIkey' : ORS_API_KEY })

# Converting distance matrix between hospitals into 2D array structure
dist_data = np.array(list(distMeters.values()))
d_size = int(math.sqrt(len(distMeters)))
d_shape = (d_size, d_size)
dist_matrix = dist_data.reshape(d_shape)

# Create safety score matrix
safety_matrix = np.zeros((len(hospital_safety), len(hospital_safety)))
for i in range(len(hospital_safety)):
    for j in range(len(hospital_safety)):
        if i == j:
            safety_matrix[i][j] = 0
        else:
            safety_matrix[i][j] = hospital_safety[i] * hospital_safety[j] * 0.01

# Constructing data structure for optimization model
n = len(hospital_list)
hospitals = range(n)
hospital = range(1, n)
dist_dict = {(i, j): dist_matrix[i][j] for i in hospitals for j in hospitals}
safety_dict = {(i, j): safety_matrix[i][j] for i in hospitals for j in hospitals}


# Parameter values
d_max = 716.742
d_min = 155.373
s_max = 2178.14
s_min = 2173.14


# Set model(distance)
md = gp.Model('Waste_VRP_distance')

# Decision Variables(distance)
y_vars = md.addVars(dist_dict.keys(), obj=dist_dict, vtype=GRB.BINARY, name='y')
u_vars = md.addVars(n)

# Constraints(distance)
md.addConstrs(quicksum(y_vars[i, j] for j in hospitals if i != j) == 1 for i in hospitals)
md.addConstrs(quicksum(y_vars[i, j] for i in hospitals if i != j) == 1 for j in hospitals)
md.addConstrs(u_vars[i] - u_vars[j] + n*y_vars[i, j] <= n-1 for i in hospital for j in hospital)
md.addConstrs(u_vars[i] <= n-1 for i in hospital)
md.addConstrs(u_vars[i] >= 0 for i in hospital)

# The objective is to minimize the total distance
md.modelSense = GRB.MINIMIZE

# Optimize model(distance)
md.optimize()


# Set model(safety)
ms = gp.Model('Waste_VRP_safety')

# Decision Variables(safety)
x_vars = ms.addVars(safety_dict.keys(), obj=safety_dict, vtype=GRB.BINARY, name='x')
u_vars = ms.addVars(n)

# Constraints(safety)
ms.addConstrs(quicksum(x_vars[i, j] for j in hospitals if i != j) == 1 for i in hospitals)
ms.addConstrs(quicksum(x_vars[i, j] for i in hospitals if i != j) == 1 for j in hospitals)
ms.addConstrs(u_vars[i] - u_vars[j] + n*x_vars[i, j] <= n-1 for i in hospital for j in hospital)
ms.addConstrs(u_vars[i] <= n-1 for i in hospital)
ms.addConstrs(u_vars[i] >= 0 for i in hospital)

# The objective is to maximize the safety scores
ms.modelSense = GRB.MAXIMIZE

# Optimize model(safety)
ms.optimize()


# Set model(fuzzy)
mf = gp.Model('Waste_VRP_fuzzy')

# Decision Variables(fuzzy)
x_vars = mf.addVars(safety_dict.keys(), vtype=GRB.BINARY, name='x')
u_vars = mf.addVars(n)
w1m_var = mf.addVar()
w1g_var = mf.addVar()
lambda_var = mf.addVar(obj=1, name='lambda')

# Constraints(fuzzy)
mf.addConstr(w1m_var == quicksum(dist_dict[i, j] * x_vars[i, j] for j in hospitals for i in hospitals if i != j))
mf.addConstr(w1g_var == quicksum(safety_dict[i, j] * x_vars[i, j] for j in hospitals for i in hospitals if i != j))

mf.addConstr(lambda_var <= (d_max - w1m_var)/(d_max - d_min))
mf.addConstr(lambda_var <= (w1g_var - s_min)/(s_max - s_min))

mf.addConstrs(quicksum(x_vars[i, j] for j in hospitals if i != j) == 1 for i in hospitals)
mf.addConstrs(quicksum(x_vars[i, j] for i in hospitals if i != j) == 1 for j in hospitals)
mf.addConstrs(u_vars[i] - u_vars[j] + n*x_vars[i, j] <= n-1 for i in hospital for j in hospital)
mf.addConstrs(u_vars[i] <= n-1 for i in hospital)
mf.addConstrs(u_vars[i] >= 0 for i in hospital)

# The objective is to maximize the general satisfaction level
mf.modelSense = GRB.MAXIMIZE

# Optimize model(fuzzy)
mf.optimize()


# Making routes for each optimization problem (Distance, Safety score, Multi-objective)
vehicle_routes = [[0], [0], [0]]
decision_lists = [[], [], []]

for i in range(len(vehicle_routes)):
    if i == 0:
        for v in md.getVars():
            if round(v.x) == 1 and v.varName.startswith('y'):
                decision_lists[i].append(eval(v.varName[1:]))
    elif i == 1:
        for v in ms.getVars():
            if round(v.x) == 1 and v.varName.startswith('x'):
                decision_lists[i].append(eval(v.varName[1:]))
    elif i == 2:
        for v in mf.getVars():
            if round(v.x) == 1 and v.varName.startswith('x'):
                decision_lists[i].append(eval(v.varName[1:]))

for i in range(len(vehicle_routes)):
    for num in range(len(decision_lists[i])):
        for num in range(len(decision_lists[i])):
            if decision_lists[i][num][0] in vehicle_routes[i] and decision_lists[i][num][1] not in vehicle_routes[i]:
                vehicle_routes[i].append(decision_lists[i][num][1])
    vehicle_routes[i].append(0)


# Redefined solution route to match the data structure of Veroviz module
solution_routes = [[], [], []]
for i in range(len(solution_routes)):
    for j in range(len(vehicle_routes[i]) - 1):
        solution_routes[i].append([vehicle_routes[i][j]+1, vehicle_routes[i][j+1]+1])

# Visualization of the final solution route
for i in range(len(vehicle_routes)):

    mySolution = {
        'VRP': solution_routes[i]
    }
    myAssignments = vrv.initDataframe('assignments')

    if i == 0:
        break

    if i == 0:
        vehicleProperties = {
                'VRP': {'model': 'veroviz/models/car_red.gltf',
                        'leafletColor': 'red'}
        }
    elif i == 1:
        vehicleProperties = {
            'VRP': {'model': 'veroviz/models/car_red.gltf',
                          'leafletColor': 'blue'}
        }
    elif i == 2:
        vehicleProperties = {
            'VRP': {'model': 'veroviz/models/car_red.gltf',
                          'leafletColor': 'green'}
        }

    for v in mySolution:
        endTimeSec = 0.0
        for arc in mySolution[v]:
            [myAssignments, endTimeSec] = vrv.addAssignment2D(
                initAssignments=myAssignments,
                objectID=v,
                modelFile=vehicleProperties[v]['model'],
                startLoc=list(myNodes[myNodes['id'] == arc[0]][['lat', 'lon']].values[0]),
                endLoc=list(myNodes[myNodes['id'] == arc[1]][['lat', 'lon']].values[0]),
                startTimeSec=endTimeSec,
                leafletColor=vehicleProperties[v]['leafletColor'],
                routeType='fastest',
                dataProvider='ORS-online',
                dataProviderArgs={'APIkey': ORS_API_KEY})
    if i == 0:
        myMap = vrv.createLeaflet(nodes=myNodes, arcs=myAssignments, mapFilename="seoul_vrp_distance.html")
    elif i == 1:
        myMap = vrv.createLeaflet(nodes=myNodes, arcs=myAssignments, mapFilename="seoul_vrp_safety.html")
    elif i == 2:
        myMap = vrv.createLeaflet(nodes=myNodes, arcs=myAssignments, mapFilename="seoul_vrp_fuzzy.html")


# Calculating total distance of the tour
def total_dist(x):
    distance = np.zeros(3)
    for j in range(len(vehicle_routes[x]) - 1):
        distance[x] += dist_matrix[vehicle_routes[x][j], vehicle_routes[x][j+1]]
    return round(distance[x], 3)


# Calculating total safety score of the tour
def total_score(x):
    safety_score = np.zeros(3)
    for j in range(len(vehicle_routes[x]) - 1):
        safety_score[x] += safety_matrix[vehicle_routes[x][j], vehicle_routes[x][j+1]]
    return round(safety_score[x], 2)


# Print solution
print("-----------------------------------------------------------------------------------------")
for i in range(len(vehicle_routes)):
    if i == 0:
        print("< Minimize total distance >")
        print('Optimal route: ', vehicle_routes[i])
        print('Total distance: ', total_dist(i), 'km')
        print('Total safety score: ', total_score(i), 'points')
        print("")
    elif i == 1:
        print("< Maximize total safety score >")
        print('Optimal route: ', vehicle_routes[i])
        print('Total distance: ', total_dist(i), 'km')
        print('Total safety score: ', total_score(i), 'points')
        print("")
    elif i == 2:
        print("< Maximize general satisfaction level >")
        print('Optimal route: ', vehicle_routes[i])
        print('Optimal lambda value: %g' % mf.objVal)
        print('Total distance: ', total_dist(i), 'km')
        print('Total safety score: ', total_score(i), 'points')
