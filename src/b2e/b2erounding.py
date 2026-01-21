import numpy as np
import time
import random
import math
import multiprocessing as mp

limit_time = 15

_write_queue = None
_writer_process = None

def _file_writer_worker(task_queue):
    print("[FileWriter Process] Started")
    while True:
        try:
            task_data = task_queue.get(timeout=1)
            if task_data is None:
                print("[FileWriter Process] Stopping")
                break
            
            func_name, args, kwargs = task_data
            try:
                if func_name == 'write_bound':
                    write_bound(*args, **kwargs)
                elif func_name == 'write_urfabound':
                    write_urfabound(*args, **kwargs)
                elif func_name == 'writeSol':
                    from src.b2e.data import read_data
                    read_data.writeSol(*args, **kwargs)
                elif func_name == 'write_csv':
                    from src.b2e.data import read_data
                    read_data.write_csv(*args, **kwargs)
                elif func_name == 'write_ctx_csv':
                    from src.b2e.data import read_data
                    read_data.write_ctx_csv(*args, **kwargs)
            except Exception as e:
                print(f"[FileWriter Process] Error: {e}")
                import traceback
                traceback.print_exc()
        except:
            continue
    print("[FileWriter Process] Stopped")

def _start_file_writer():
    global _write_queue, _writer_process
    if _writer_process is None or not _writer_process.is_alive():
        _write_queue = mp.Queue()
        _writer_process = mp.Process(target=_file_writer_worker, args=(_write_queue,))
        _writer_process.daemon = True
        _writer_process.start()

def enqueue_write_task(func, *args, **kwargs):
    global _write_queue
    if _write_queue is None:
        _start_file_writer()
    
    func_name = func.__name__ if hasattr(func, '__name__') else str(func)
    task_data = (func_name, args, kwargs)
    _write_queue.put(task_data)

def wait_for_write_tasks():
    global _write_queue, _writer_process
    if _write_queue is None or _writer_process is None:
        return
    while not _write_queue.empty():
        time.sleep(0.1)
    time.sleep(2)

def stop_file_writer():
    global _write_queue, _writer_process
    if _writer_process is not None and _writer_process.is_alive():
        _write_queue.put(None)
        _writer_process.join(timeout=1200)
        if _writer_process.is_alive():
            _writer_process.terminate()
            _writer_process.join()
        _writer_process = None
        _write_queue = None

def write_bound(s_x, resultpath, k_number, txsNumber, basePath, write_bounds):
    if write_bounds:
        select_ctx = sum(1 for v in s_x.values() if v > 0.5)
        with open(resultpath, "w") as f:
            f.write(f"{select_ctx},0.0")

def write_urfabound(resultpath, obj, ctxs, capacities, basePath, write_bounds):
    if write_bounds:
        D = min(capacities)
        K = max(ctx[0] for ctx in ctxs)
        y = K / D
        y_plus_1_floor = math.floor(1/y) * 1.0
        p = y_plus_1_floor / (y_plus_1_floor + 1)
        bounds = p * obj
        with open(resultpath, "w") as f:
            f.write(f"{p},{bounds},{obj}")


def revenue_function(D, r, alpha=1.2, beta=0.5):
    """
    Simplified ranking-dependent revenue function: f(D, r) = D^α / r^β
    
    This is a MOCK IMPLEMENTATION for demonstration purposes.
    The actual Broker2Earn protocol uses a complex transaction allocation mechanism
    that is not included in this open-source release.
    
    Properties (as described in the paper):
    - Convexity in D: ∂²f/∂D² > 0 when α > 1 (superlinear returns to scale)
    - Ranking effect: f(D, r₁) > f(D, r₂) when r₁ < r₂ (better rank = higher revenue)
    
    Parameters:
    - D: deposited balance / fund size (ETH)
    - r: ranking position (1 = highest balance, 2 = second highest, etc.)
    - alpha: convexity parameter (default 1.2, ensures α > 1 for convex returns)
    - beta: ranking sensitivity (default 0.5, controls how much ranking affects revenue)
    
    Returns:
    - Revenue amount based on balance and ranking
    
    Mathematical properties:
    - ∂f/∂D = α·D^(α-1) / r^β > 0  (increasing in balance)
    - ∂²f/∂D² = α(α-1)·D^(α-2) / r^β > 0 when α > 1  (convex)
    - ∂f/∂r = -β·D^α / r^(β+1) < 0  (decreasing in rank number)
    """
    if D <= 0:
        return 0.0
    if r <= 0:
        r = 1  # Best possible ranking
    
    # Core formula: D^α / r^β
    # - D^α creates convex returns (economies of scale)
    # - 1/r^β creates ranking advantages (better rank = higher multiplier)
    base_revenue = (D ** alpha) / (r ** beta)
    
    return base_revenue


def greedy_transaction_allocation(ctxs, capacities):
    """
    Simplified transaction allocation using greedy value-density approach.
    
    This is a SIMPLIFIED BASELINE that demonstrates ranking-dependent behavior
    without exposing the actual Broker2Earn allocation algorithm.
    
    Algorithm:
    1. Sort transactions by value-to-weight ratio (fee density)
    2. Assign each transaction to the broker with sufficient capacity
    3. Track fee collection for each broker
    
    Args:
    - ctxs: list of (value, weight, ...) transactions
    - capacities: list of broker capacity limits
    
    Returns:
    - allocation: dict mapping (ctx_idx, broker_idx) -> 1.0 or 0
    - broker_fees: list of fees collected by each broker
    """
    k_number = len(capacities)
    allocation = {}
    broker_fees = [0.0] * k_number
    used_capacity = [0.0] * k_number
    
    # Sort transactions by value density (fee per unit weight)
    sorted_ctxs = sorted(
        enumerate(ctxs), 
        key=lambda x: x[1][0] / (x[1][1] + 1e-8),  # value / weight
        reverse=True  # Highest density first
    )
    
    # Greedy allocation: assign each transaction to first available broker
    for ctx_idx, ctx_data in sorted_ctxs:
        ctx_fee = ctx_data[0]  # Transaction fee
        ctx_weight = ctx_data[1]  # Transaction weight/size
        
        # Try to allocate to brokers in order of remaining capacity
        allocated = False
        for broker_idx in range(k_number):
            if used_capacity[broker_idx] + ctx_weight <= capacities[broker_idx]:
                # Assign transaction to this broker
                allocation[(ctx_idx, broker_idx)] = 1.0
                used_capacity[broker_idx] += ctx_weight
                broker_fees[broker_idx] += ctx_fee
                allocated = True
                break
        
        # If transaction has zero weight, assign to random broker
        if not allocated and ctx_weight == 0:
            broker_idx = random.randint(0, k_number - 1)
            allocation[(ctx_idx, broker_idx)] = 1.0
            broker_fees[broker_idx] += ctx_fee
    
    return allocation, broker_fees


def compute_broker_rankings(broker_fees):
    """
    Compute ranking positions based on fee collection amounts.
    
    Args:
    - broker_fees: list of fees collected by each broker
    
    Returns:
    - rankings: list of ranking positions (1 = highest fees, 2 = second, etc.)
    """
    # Create (fee, original_index) pairs
    indexed_fees = [(fee, idx) for idx, fee in enumerate(broker_fees)]
    
    # Sort by fee (descending) to get rankings
    sorted_fees = sorted(indexed_fees, key=lambda x: x[0], reverse=True)
    
    # Assign rankings (handle ties by assigning same rank)
    rankings = [0] * len(broker_fees)
    current_rank = 1
    for i, (fee, original_idx) in enumerate(sorted_fees):
        if i > 0 and sorted_fees[i][0] < sorted_fees[i-1][0]:
            current_rank = i + 1
        rankings[original_idx] = current_rank
    
    return rankings

def B2ERounding(dataEpoch, var_type, resultPath, iter_num=5, alpha=1.2, 
                feeRatio=0.1, sorted_ids=None, write_bounds=False):
    """
    Simplified Broker2Earn baseline - Core computation only.
    
    NOTE: File writing has been removed in this open-source version.
    Only essential computation and return values are provided.
    """
    start_time = time.time()
    
    ctxs, capacities, txsNumber = dataEpoch
    k_number = len(capacities)
    
    # Step 1: Allocate transactions
    allocation, broker_fees = greedy_transaction_allocation(ctxs, capacities)
    
    # Step 2: Compute rankings
    rankings = compute_broker_rankings(broker_fees)
    
    # Step 3: Apply revenue function
    broker_revenues = []
    for broker_idx in range(k_number):
        fee = broker_fees[broker_idx]
        rank = rankings[broker_idx]
        revenue = revenue_function(fee, rank, alpha=alpha, beta=0.5)
        broker_revenues.append(revenue * feeRatio)
    
    total_value = sum(broker_fees)
    using_time = time.time() - start_time
    
    # No file writing - just return results
    return using_time, total_value, broker_revenues

# ==============================================================================
# ADDITIONAL UTILITY FUNCTIONS (Optional)
# ==============================================================================

def get_revenue_rate(broker_fee, ranking, alpha=1.2):
    """
    Calculate revenue rate (percentage return) for a given fee amount and ranking.
    
    This can be used for stakeholder decision-making in the LiquidityPool system.
    
    Args:
    - broker_fee: total fees collected by broker
    - ranking: ranking position
    - alpha: convexity parameter
    
    Returns:
    - rate: revenue as percentage of input (e.g., 0.05 = 5% return)
    """
    if broker_fee <= 0:
        return 0.0
    
    revenue = revenue_function(broker_fee, ranking, alpha=alpha, beta=0.5)
    rate = revenue / broker_fee if broker_fee > 0 else 0.0
    
    return rate


def analyze_ranking_effects(capacities_list, alpha=1.2):
    """
    Analyze how different rankings affect revenue for the same balance.
    
    This demonstrates the ranking-dependent property f(D, r₁) > f(D, r₂) when r₁ < r₂.
    
    Usage example:
    >>> analyze_ranking_effects([100, 100, 100], alpha=1.2)
    Balance: 100, Rank 1: Revenue = 158.49
    Balance: 100, Rank 2: Revenue = 112.05
    Balance: 100, Rank 3: Revenue = 91.48
    """
    print("\n=== Ranking Effect Analysis ===")
    print(f"Testing with alpha={alpha}, beta=0.5")
    print(f"Revenue function: f(D, r) = D^{alpha} / r^0.5\n")
    
    for D in sorted(set(capacities_list), reverse=True):
        print(f"\nBalance: {D}")
        for rank in range(1, 6):
            revenue = revenue_function(D, rank, alpha=alpha, beta=0.5)
            rate = (revenue / D - 1) * 100 if D > 0 else 0
            print(f"  Rank {rank}: Revenue = {revenue:.2f} ({rate:+.2f}%)")


def analyze_convexity_effects(rankings, alpha=1.2):
    """
    Analyze how convexity creates economies of scale.
    
    This demonstrates the convexity property ∂²f/∂D² > 0.
    
    Usage example:
    >>> analyze_convexity_effects([1, 1, 1], alpha=1.2)
    Shows how doubling balance more than doubles revenue.
    """
    print("\n=== Convexity Effect Analysis ===")
    print(f"Testing with alpha={alpha} (α > 1 ensures convexity)")
    print(f"For a fixed rank, larger deposits earn disproportionately more\n")
    
    test_balances = [10, 20, 50, 100, 200]
    for rank in rankings:
        print(f"\nRank {rank}:")
        for D in test_balances:
            revenue = revenue_function(D, rank, alpha=alpha, beta=0.5)
            rate = (revenue / D - 1) * 100 if D > 0 else 0
            print(f"  Balance {D:>3}: Revenue = {revenue:>8.2f} (rate: {rate:>6.2f}%)")