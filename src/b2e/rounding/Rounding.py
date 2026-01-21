import numpy as np
import random


def filter_ctx(weights_capacities, capacities, new_ctxs, weights):
    res = []
    for index, val in enumerate(new_ctxs):
        if(weights_capacities + weights[val] < capacities):
            res.append(val)
    return res

def b2e_rounding(s_x, k_number, ctxs, capacities):
    ctx_number = len(ctxs)
    weights = [item[1] for item in ctxs]
    values = [item[0] for item in ctxs]
    n_x = np.zeros((k_number, ctx_number), dtype = np.int32)
    ori_x = np.zeros((k_number, ctx_number), dtype = np.float32)
    
    keys = list(s_x.keys())
    # ori_x [CAPACITY_ID, CTX_ID]
    for key in keys:
        ori_x[key[1], key[0]] = s_x[key]
            
    weights_capacities = [0 for i in capacities]
    # print(ctx_number, k_number, len(x))
    value = 0
    ctx_selected = [0 for i in range(ctx_number)]
    for k in range(k_number):
        new_ctxs = []
        p = np.array([])
        for index, val in enumerate(ori_x[k, :]):
            if(ctx_selected[index] < 1 and val > 0.0 and weights[index] < capacities[k]):
                new_ctxs.append(index)
                p = np.append(p, val)
        # print("new_ctxs ", new_ctxs)
        p = p / sum(p)
        # print("ctxs ", new_ctxs)
        # print("ctxs weight ", [weights[i] for i in new_ctxs])
        # print("p is ", p)
        # print(len(new_ctxs))
        while(len(new_ctxs) != 0):
            # print(k, len(new_ctxs))
            # np.random.seed(0)
            ctx_id = np.random.choice(new_ctxs, p = p)
            if(weights_capacities[k] + weights[ctx_id] <= capacities[k]):
                weights_capacities[k] += weights[ctx_id]
                value += values[ctx_id]
                
            n_x[k, ctx_id] = 1
            ctx_selected[ctx_id] = 1
            
            pos = new_ctxs.index(ctx_id)
            new_ctxs.pop(pos)
            new_ctxs = filter_ctx(weights_capacities[k], capacities[k], new_ctxs, weights)
            p = np.array([ori_x[k, i] for i in new_ctxs])
            # print("p0 ", p)
            p = p / sum(p)
            # print("p1 ", p)
        # print("===========================")    
            # print("k is ", k, len(new_ctxs), len(p), weights_capacities[k], capacities[k])
    # print("The sum is ", n_x.sum())
    # print("=" * 50)
    print(value)
    return n_x


# print(pipage_rounding([0.1,0.8,0.4,0.7]))

# epoch = 10000
# x = [0.1,0.9,0.3,0.7,0.4,0.6]
# cnt = np.zeros(len(x))
# for i in range(epoch):
    # x_round = pipage_rounding(x)
    # print(x_round)
    # cnt += x_round

# print(cnt/epoch)
# print(x_round)
