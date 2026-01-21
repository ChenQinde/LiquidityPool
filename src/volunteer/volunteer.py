import random
from typing import List, Dict
from src.utils.logger import Logger

class Volunteer:
    def __init__(self, id: int, initial_balance: int, risk_tolerance: float = 0.5, acc: bool = False):
        self.id = id
        self.logger = Logger.get_logger()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_tolerance = risk_tolerance
        self.current_liquiditypool = None
        self.earnings_history = []
        self.earnings_rate_history = []
        self.b2e_rate_history = [] # Added this attribute to store previous B2E return rate
        self.b2e_revenue_history = [] # Added this attribute to store previous B2E revenue
        self.acc = acc

        # Set current number of rounds elapsed
        self.iteration = 0

        # When was the last time leaving liquiditypool
        self.last_leave_iteration = -100

        self.stay_intervals = [3,6]

        # After joining liquiditypool, how many rounds must pass before being able to leave, randomly set an integer between 3 and 6
        self.leave_interval = random.randint(self.stay_intervals[0], self.stay_intervals[1])
    
    def make_decision(self, market_data: Dict, liquiditypools: List[Dict]) -> Dict:
        """
        Make decisions based on market data and all LiquidityPool information
        """
        # Increment by 1 each time
        self.iteration = self.iteration + 1
        # Calculate expected return rate from direct B2E participation
        b2e_rate = self.b2e_rate_history[-1]
        # self.logger.info(f"volunteer {self.id} b2e : {b2e_rate}")
        best_lp_list = []
        best_lp_rate = 0

        self.logger.info(f"volunteer {self.id} b2e_rate : {b2e_rate}")
        for lp in liquiditypools:
            # Add random fluctuation to prevent volunteers from joining one pool en masse when liquiditypool1 and 2 are nearly identical
            current_lp_rate = lp.get_user_earnings_rate() * random.uniform(0.95,1.05)
            # if self.id % 10 ==0:
            # self.logger.info(f"volunteer {self.id} {lp.id} : {current_lp_rate}")
            # self.logger.info(f"b2e rate is:{b2e_rate}")
            self.logger.info(f"volunteer {self.id} {lp.id} : {current_lp_rate}")
            
            if current_lp_rate > best_lp_rate:
                best_lp_rate = current_lp_rate
                best_lp_list = [lp]

            # elif abs(current_lp_rate-best_lp_rate) / current_lp_rate < 0.01:
            #     best_lp_list.append(lp)
        
        best_lp = None
        if best_lp_list:
            best_lp = random.choice(best_lp_list)
        
        if best_lp_rate > b2e_rate:

            if(not self.current_liquiditypool):
                # Join liquiditypool from b2e, update self.last_leave_iteration
                self.last_leave_iteration = self.iteration
                return {'action': 'join', 'liquiditypool_id': best_lp.id, "leave_id": -1}
          
            if best_lp.id != self.current_liquiditypool.id:
                # If stay duration is too short, force to stay
                if self.iteration - self.last_leave_iteration < self.leave_interval:
                    return {'action': 'stay'}
                
                # Join another liquiditypool, update self.last_leave_iteration
                self.last_leave_iteration = self.iteration
                self.leave_interval = random.randint(self.stay_intervals[0], self.stay_intervals[1])
                return {'action': 'join', 'liquiditypool_id': best_lp.id, "leave_id": self.current_liquiditypool.id}
                
            elif best_lp.id == self.current_liquiditypool.id:
                return {'action': 'stay'}
        else:
            if(not self.current_liquiditypool):
                return {'action': 'leave',"liquiditypool_id":-1, "leave_id":-1}
            return {'action': 'leave', 'liquiditypool_id': -1, "leave_id": self.current_liquiditypool.id}

    def update_market(self, b2e_result: Dict, liquiditypools: List[Dict]): 
        if(self.current_liquiditypool != None):
            self.earnings_rate_history.append(self.current_liquiditypool.get_user_earnings_rate())
            self.earnings_history.append(self.earnings_rate_history[-1] * self.balance)
        elif(self.current_liquiditypool == None):
            self.earnings_history = [b2e_result["earnings"][self.id]]
            self.earnings_rate_history = [b2e_result["earnings"][self.id] / self.balance]
            self.b2e_rate_history = [b2e_result["rates"][self.id]]
            self.b2e_revenue_history = [b2e_result["earnings"][self.id]]
        
    def update_decision(self, des: List[Dict], liquiditypools: List[Dict]):
        """
        Update Volunteer status
        """
        
        if(des["action"] == "leave"):
            if(des["leave_id"] != -1):
                self.current_liquiditypool.remove_user(self.id, self.balance)
                self.current_liquiditypool = None
            return self.current_liquiditypool
        
        
        
        if(des["action"] == "join"):
            current_lp = next(lp for lp in liquiditypools if lp.id == des["liquiditypool_id"])
            self.current_liquiditypool = current_lp
            if(des["leave_id"] == -1):
                current_lp.add_user(self.id, self.balance)
            else:
                leave_lp = next(lp for lp in liquiditypools if lp.id == des["leave_id"])
                leave_lp.remove_user(self.id, self.balance)  
                if self.acc:
                    self.balance += self.earnings_history[-1]
                current_lp.add_user(self.id, self.balance)  
                
        return self.current_liquiditypool

    def get_state(self) -> Dict:
        """
        Return Volunteer's current state
        """
        return {
            'id': self.id,
            'balance': self.balance,
            'current_liquiditypool': self.current_liquiditypool.id if self.current_liquiditypool != None else None,
            'total_earnings': sum(self.earnings_history),
            'last_b2e_rate': self.b2e_rate_history[-1],
            'revenue_rate': self.earnings_rate_history[-1]
        }