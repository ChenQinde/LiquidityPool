import numpy as np
from typing import List, Dict, Tuple
from src.utils.logger import Logger
import numpy as np
from typing import List, Dict, Tuple
from src.utils.logger import Logger

class ManagementFeeOptimizer:
    def __init__(self, config: Dict):
        # Basic parameters
        self.id = config["params"]["id"]
        self.initial_funds = config["params"]["initial_funds"]
        self.min_fee_rate = 0.05  # 5% minimum fee rate
        self.max_fee_rate = 0.99  # 99% maximum fee rate
        self.base_fee_rate = 0  # 0% base fee rate
        self.logger = Logger.get_logger()

        # Current state and historical records
        self.current_fee_rate = config["params"]["initial_tax_rate"]
        self.fee_rate_history = [self.current_fee_rate]
        self.revenue_history = []
        self.net_revenue_history = []
        self.funds_history = []
        self.invested_funds_history = []  # Added: record investor fund history
        
        self.window_size = 5  # Window size for calculating moving average
        self.adjust_threshold = 0.1  # Revenue change threshold
        
        self.historical_max_successful_rate = self.max_fee_rate  # Added: record historical safe fee rate upper limit
        self.success_threshold = 5  # Added: consecutive success count threshold
        self.failure_threshold = 2  # Added: consecutive failure count threshold
        self.consecutive_success = 0  # Added: consecutive success counter
        self.consecutive_failure = 0  # Added: consecutive failure counter
        self.status = 0
        self.previous_invested_funds = None  # Added: previous round investor funds
        

    def calculate_adjustment_coefficient(self, total_funds: float, competitor_funds: float, initial_funds: float) -> float:
        """
        Calculate adjustment coefficient based on the effective fund difference multiple with competitors
        
        Args:
            total_funds: Total funds of current LiquidityPool
            competitor_funds: Total funds of competitor LiquidityPool
            initial_funds: Initial funds of LiquidityPool
            
        Returns:
            float: Adjustment coefficient (between 0.05-0.2)
        """
        # Calculate actual investor invested funds
        invested_funds = total_funds - initial_funds
        # Assume competitor also has the same initial_funds, calculate their investor funds
        competitor_invested_funds = competitor_funds - initial_funds
        
        if invested_funds <= 0:
            return 0.2  # If no investor funds, use maximum adjustment coefficient
            
        # Calculate effective fund difference multiple
        funds_ratio = competitor_invested_funds / invested_funds if invested_funds > 0 else float('inf')
        
        # Set adjustment coefficient based on multiple difference
        if funds_ratio > 2:  # Competitor investor funds exceed 2x
            return 0.2
        elif funds_ratio > 1.5:  # Competitor investor funds exceed 1.5x
            return 0.15
        elif funds_ratio > 1.2:  # Competitor investor funds exceed 1.2x
            return 0.1
        else:  # Fund gap is relatively small
            return 0.05
            
    def update_rate_bounds(self, net_revenue: float):
        """
        Update fee rate upper limit based on investor fund changes
        """
        if len(self.net_revenue_history) <= 5 or len(self.invested_funds_history) <= 5 or len(self.fee_rate_history) <= 5:
            return
            
       
        # Calculate revenue trend
        recent_revenues = self.net_revenue_history[-5:]
        x = np.arange(5)
        revenue_slope = np.polyfit(x, recent_revenues, 1)[0]
        revenue_change = revenue_slope / max(abs(np.mean(recent_revenues)), 1)
        
        # Calculate fund trend
        recent_funds = self.invested_funds_history[-5:]
        funds_slope = np.polyfit(x, recent_funds, 1)[0]
        funds_change = funds_slope / max(abs(np.mean(recent_funds)), 1)
        
        # Calculate fee rate trend
        recent_rates = self.fee_rate_history[-5:]
        rate_slope = np.polyfit(x, recent_rates, 1)[0]
        rate_change = rate_slope / max(abs(np.mean(recent_rates)), 1)
    
    

        # print(self.id, " ", funds_change)
        # self.logger.info(f"LiquidityPool {self.id} - funds_change: {funds_change}")
            
        # Record diagnostic information
        self.logger.info(f"LiquidityPool {self.id} - Diagnostics: revenue_change={revenue_change:.6f}, " +
                        f"funds_change={funds_change:.6f}, rate_change={rate_change:.6f}, consecutive_success={self.consecutive_success}")
    
        if(self.status == 0 and revenue_change != 0.0 and abs(funds_change) <= 1e-10):
            self.consecutive_failure = 0
            self.consecutive_success = 0
            if(revenue_change < 0):
                self.historical_max_successful_rate = max(
                    self.historical_max_successful_rate,
                    self.current_fee_rate * 1.0001
                )
            if(revenue_change > 0):
                self.historical_max_successful_rate = max(
                    self.historical_max_successful_rate,
                    self.current_fee_rate * 0.9999
                )
            self.status = 1
            self.logger.info(f"LiquidityPool {self.id} - Market saturation persists considering stable rates: {self.historical_max_successful_rate:.4f}")

            return 1
        if(self.status == 1 and funds_change <= 0):
            if(revenue_change > 0):
                self.status = 2
                self.optimal_rate_lower = self.current_fee_rate
                self.optimal_rate_upper = self.current_fee_rate
                self.logger.info(f"LiquidityPool {self.id} - Status 2: Revenue increasing despite fund decrease, optimal rate may be {self.current_fee_rate:.4f}")

            if(revenue_change <= 0):
                self.status = 3
                self.optimal_rate_lower = min(self.current_fee_rate, self.fee_rate_history[-2])
                self.optimal_rate_upper = max(self.current_fee_rate, self.fee_rate_history[-2])
                self.logger.info(f"LiquidityPool {self.id} - Status 3: Optimal rate between {self.optimal_rate_lower:.4f} and {self.optimal_rate_upper:.4f}")
            return 1
        elif(self.status == 1 and funds_change > 0):
            self.logger.info(f"LiquidityPool {self.id} - Status reset: fund increase, not optimal fee.")
            self.status = 0
            
            
        # Update consecutive success/failure count
        if revenue_change >= 1e-5 or funds_change >= 1e-5:  
            self.consecutive_success += 1
            self.consecutive_failure = 0
            
            # If consecutive successes are sufficient, consider current fee rate as safe
            if self.consecutive_success >= self.success_threshold:
                self.historical_max_successful_rate = max(
                    self.historical_max_successful_rate,
                    self.current_fee_rate
                )
                self.logger.info(f"LiquidityPool {self.id} - Ideal growth, updating max successful rate to: {self.historical_max_successful_rate:.4f}")
    
                # self.consecutive_success = 0
          
        # elif revenue_change >= 1e-5 and funds_change < 0:
            # Revenue growing but funds decreasing: may indicate fee rate adjustment is effective but too high
            # Maintain status quo, do not increase success or failure count
            # self.logger.info(f"LiquidityPool {self.id} - Revenue growing but losing funds, maintaining current strategy")
          
        elif revenue_change <= -1e-5:  # 1% negative growth is considered failure
                
                
            self.consecutive_failure += 1
            self.consecutive_success = 0
            
            # If consecutive failures, may need to adjust historical maximum safe fee rate
            if self.consecutive_failure >= self.failure_threshold:
                self.historical_max_successful_rate = min(
                    self.historical_max_successful_rate,
                    self.current_fee_rate * 0.95  # Slightly reduce historical maximum safe fee rate
                )
                # self.consecutive_failure = 0
                
                self.logger.info(f"LiquidityPool {self.id} - decrease, updating max successful rate to: {self.historical_max_successful_rate:.4f}")
        
    def is_liquiditypool_id(self, id: int) -> bool:
        """Determine if it is a LiquidityPool ID"""
        return str(id).startswith('LiquidityPool')

    def optimize(self, iteration: int, last_b2e_rates: Dict[int, float], 
                last_b2e_earnings: Dict[int, float], participation_rate1: float,
                total_investment: float, current_funds: Dict[int, float],
                current_earn: float, transaction_data: List[Tuple[int, int, str, str]]) -> float:
        """Optimization function"""
        
        
        
        # Update historical data
        current_own_funds = current_funds[self.id]
        current_invested_funds = current_own_funds - self.initial_funds
        
        
        previous_invested_funds = self.invested_funds_history[-1] if len(self.invested_funds_history) > 0 else 0
        is_losing_investors = current_invested_funds < previous_invested_funds * 0.9  # Investors decreased by more than 10%

        # If investors are leaving in large numbers, immediately adjust historical_max_successful_rate
        if is_losing_investors and previous_invested_funds > 0:
            # Adjust historical safe fee rate to 80% of current rate to ensure it won't rise too high next time
            self.historical_max_successful_rate = min(
                self.historical_max_successful_rate,
                self.fee_rate_history[-2] * 0.8
            )
            self.status = 0
            self.logger.info(f"LiquidityPool {self.id} - Investors leaving, adjusting historical_max_successful_rate rate to: {self.historical_max_successful_rate:.4f}")
        
        
        
        
        
        self.revenue_history.append(current_earn)
        self.funds_history.append(current_own_funds)
        self.invested_funds_history.append(current_invested_funds)
        
        # Calculate net revenue
        net_revenue = current_earn * (current_invested_funds) / current_own_funds * self.current_fee_rate
        self.net_revenue_history.append(net_revenue)
        
        # if(net_revenue >= 0 and self.historical_max_successful_rate == -1):
            # self.historical_max_successful_rate = self.current_fee_rate
        
        
        
        
        
        # Update fee rate range
        self.update_rate_bounds(net_revenue)
        
        
    # Added: If no investors, proactively reduce fee rate
        if current_invested_funds <= 0:
            
            self.consecutive_success = 0
            self.consecutive_failure = 0
            

            if(previous_invested_funds != 0 and len(self.fee_rate_history) >= 2):
                self.current_fee_rate = min(self.fee_rate_history[-2] * 1.05, self.max_fee_rate)
                self.fee_rate_history.append(self.current_fee_rate)
                self.logger.info(f"LiquidityPool {self.id} - No investors, and last epoch has investor, return to last fee rate: {self.current_fee_rate:.4f}")
                return self.current_fee_rate
            
            # Gradually reduce fee rate based on iteration count
            # no_investor_adjustment = -0.01 * min(iteration, 10) / 10  # Reduce by up to 1 percentage point
            no_investor_adjustment = - self.current_fee_rate / 2  # Reduce by up to half
            
            # Calculate new fee rate, but ensure it's not lower than minimum fee rate
            new_fee_rate = max(self.min_fee_rate, self.current_fee_rate + no_investor_adjustment)
            
            self.current_fee_rate = new_fee_rate
            self.fee_rate_history.append(self.current_fee_rate)
            
            self.logger.info(f"LiquidityPool {self.id} - No investors, reducing fee rate: {self.current_fee_rate:.4f}")
            
            return self.current_fee_rate
        
        
        # Collect data in the first two rounds
        if iteration <= 2:

            return self.current_fee_rate
            
        
        # Get competitor data
        competitor_funds = {k: v for k, v in current_funds.items() 
                          if str(k).startswith('LiquidityPool') and k != self.id}
        
        if not competitor_funds:
            return self.current_fee_rate
           



           
        # Find the competitor with the most funds
        max_competitor_id = max(competitor_funds.items(), key=lambda x: x[1])[0]
        max_competitor_funds = competitor_funds[max_competitor_id]
        competitor_invested_funds = max_competitor_funds - self.initial_funds
        
        # Calculate fund gap ratio with competitor
        funds_diff_ratio = (competitor_invested_funds - current_invested_funds) / max(competitor_invested_funds, 1e18)
        
        # Get adjustment coefficient
        adjustment_coef = self.calculate_adjustment_coefficient(
            current_own_funds,
            max_competitor_funds,
            self.initial_funds
        )
        
        # Calculate competition-based fee rate adjustment
        if funds_diff_ratio > 0:  # Lagging situation
            # Check competitor return rate
            competitor_rate = last_b2e_rates[max_competitor_id]
            own_rate = last_b2e_rates[self.id]
            
            if own_rate < competitor_rate:
                # Return rate also lagging, need more aggressive fee rate reduction
                competition_adjustment = -adjustment_coef * funds_diff_ratio
            else:
                # Return rate leading, moderate adjustment
                competition_adjustment = -adjustment_coef * funds_diff_ratio * 0.5
        else:  # Leading situation
            # Can appropriately increase fee rate
            competition_adjustment = adjustment_coef * abs(funds_diff_ratio) * 0.3



        # bounds = (0.9 if iteration <= 50 else
                 # 50/iteration if iteration > 500 else
                 # max(0.1, 0.9 - ((iteration - 100) // 50) * 0.1))     
        # Set parameters
        rapid_decay_threshold = 100  # Rapid decay threshold
        initial_bound = 0.9          # Initial bounds value
        min_bound = 0.1              # Minimum value for linear decay
        
        # bounds calculation function
        def calculate_bounds(iteration):
            if iteration > rapid_decay_threshold:
                # Rapid decay after exceeding threshold
                return (0.1 * rapid_decay_threshold)/iteration
            else:
                # Linear decay starting from 0 (from 0.9 down to 0.1)
                decay_rate = (initial_bound - min_bound) / rapid_decay_threshold
                return initial_bound - iteration * decay_rate
                
        bounds = calculate_bounds(iteration)
                 
        # Calculate upper and lower bound range
        # lower_bound = max(self.min_fee_rate, self.historical_max_successful_rate * ( 1 - bounds))  # Lower bound not below min_fee_rate
        lower_bound = self.min_fee_rate  # Lower bound not below min_fee_rate
        
        upper_bound = min(self.max_fee_rate, self.historical_max_successful_rate * ( 1 + bounds))  # Upper bound not exceeding max_fee_rate
        if(self.status >= 2):
            lower_bound = self.optimal_rate_lower
            upper_bound = self.optimal_rate_upper    

        # Apply adjustment within the range
        new_fee_rate = max(lower_bound, min(upper_bound, self.current_fee_rate + competition_adjustment))
        
        # Update current fee rate
        self.current_fee_rate = new_fee_rate
        self.fee_rate_history.append(self.current_fee_rate)
        
        self.logger.info(f"LiquidityPool {self.id} - New fee rate: {self.current_fee_rate:.4f}, "
                        f"Funds diff ratio: {funds_diff_ratio:.4f}, "
                        f"Current invested funds: {current_invested_funds}, "
                        f"Current min_fee_rate: {self.min_fee_rate}, "
                        f"Current max_fee_rate: {self.max_fee_rate}, "
                        f"Current historical_max_successful_rate: {self.historical_max_successful_rate}, "
                        f"Competitor invested funds: {competitor_invested_funds}")
        return self.current_fee_rate