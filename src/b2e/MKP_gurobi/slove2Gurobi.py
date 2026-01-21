import time
from src.b2e.data import read_data
from src.b2e.utils import utils

# SCALE = 1e16
SCALE = 1

def read_example(path):
    with open(path, "r") as f:
        allData = f.readlines()
        slice = allData[1].strip().split(" : ")[1].split(",")
        txsNumber = int(slice[1]) - int(slice[0])
        data = allData[3:]
        items = []
        capacities = [int(capacity) for capacity in data[1].split(", ")]
        for item in data[0].strip().split(" "):
            item = item[1:-1].split(",")
            items.append((int(item[0]), int(item[1]), item[2], item[3]))
    return items, capacities, txsNumber

def value_desity_first_linear(dataEpoch, var_type="BINARY", alpha=1, sigma=0.1):
    start_time = time.time()
          
    items, capacities, txsNumber = dataEpoch
    
    read_time = time.time() - start_time
    
    items = [[i[0] / SCALE, i[1] / SCALE, i[2]] for i in items]
    capacities = [i / SCALE for i in capacities]
    
    item_number = len(items)
    k_number = len(capacities)

    new_items = [[i, (alpha * items[i][0] * sigma + 1), items[i][1]] for i in range(len(items))]
    new_items.sort(key=lambda x: x[1]/(x[2] + 1e-6), reverse=True)
    weight_capacities = [0] * k_number
    
    start_time = time.time()
    s_x = {}
    num = 0
    value = 0
    
    for item in new_items:
        if num >= k_number:
            break
            
        if item[2] >= capacities[num]:
            continue
        
        if weight_capacities[num] < capacities[num] and weight_capacities[num] + item[2] >= capacities[num]:
            sub = capacities[num] - weight_capacities[num]
            s_x[(item[0], num)] = sub / item[2]
            weight_capacities[num] = capacities[num]
            num += 1
            if num >= k_number:
                break
            for j in range(num, k_number):
                if weight_capacities[j] + item[2] - sub <= capacities[j]:
                    weight_capacities[j] += item[2] - sub
                    s_x[(item[0], j)] = 1 - s_x[(item[0], num - 1)]
                    break
        elif weight_capacities[num] >= capacities[num]:
            continue
        else:
            weight_capacities[num] += item[2]
            s_x[(item[0], num)] = 1.0
            value += item[1]
            
    for i in new_items:
        for k in range(k_number):
            if (i[0], k) not in s_x:
                s_x[(i[0], k)] = 0.0
    
    cpu_time = time.time() - start_time
    
    return s_x, value, cpu_time, [k_number, txsNumber, items, capacities, "OPTIMAL"]