import time
import random
from itertools import permutations

base_dir = "./example/"
example_name = "medium"

item_number = 2000
k_number = 50
upppr_bound = 100
lower_bound = 50
items_weight = [random.choices(range(1, upppr_bound),k = item_number)][0]
items_value  = [random.choices(range(1, upppr_bound),k = item_number)][0]
 
print(items_weight)
print(items_value)

capacitys = [random.choices(range(upppr_bound, upppr_bound * 2),k = k_number)][0]
max_capacity = max(capacitys)

print(capacitys, max_capacity)

items = []
# (value, weight)
for i in range(len(items_weight)):
    if(items_weight[i] > max_capacity):
        continue
    items.append((items_value[i],items_weight[i]))

print(items)

res = " ".join(["(" + str(item[0]) + "," + str(item[1]) + ")" for item in items])

# for item in items:
    # res += "(" + str(item[0]) + "," + str(item[1]) + ")"
    # print(item)

print(res)

def write_to_file(path, example_name, res, capacitys):
    with open(path + example_name + ".txt", "w") as f:
        f.write(res + "\n" + str(capacitys[0]))
        for capacity in capacitys[1:]:
            f.write(", " + str(capacity))
        
        
write_to_file(base_dir, example_name, res, capacitys)