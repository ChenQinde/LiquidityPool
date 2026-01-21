import numpy as np
from typing import List, Dict
from src.liquiditypool.management_optimizer import ManagementFeeOptimizer
from src.utils.logger import Logger

class LiquidityPool:
    def __init__(self, config: Dict):
        self.id = config['id']
        
        self.logger = Logger.get_logger()
        self.initial_funds = config['initial_funds']
        self.current_funds = self.initial_funds
        self.tax_rate = config["optimizer"]["params"]['initial_tax_rate']
        self.optimizer_type = config["optimizer"]["type"]
        if(self.optimizer_type == "manage"):
            self.tax_optimizer = ManagementFeeOptimizer(
                {**config["optimizer"], 'experiment_name': config['experiment_name']})
        self.users = set()
        
        # Revenue obtained directly from b2e
        self.b2e_revenue_history = []
        self.b2e_rate_history = []
        
        # Revenue without deducting the only volunteer
        self.revenue_history = []
        
        # Revenue after deducting the only volunteer
        self.net_revenue_history = []
        self.tax_rate_history = []
        self.current_funds_history = []
        self.total_user_funds = 0
        self.iteration=0
        # self.historical_max_successful_rate_history = []  
        
    def get_earnings_rank(self, b2e_result: Dict) -> int:
        # Get all earnings data
        earnings = b2e_result["earnings"]

        # Get the earnings for the current id
        current_earnings = earnings[self.id]

        # Sort earnings data in descending order to get a sorted list
        sorted_earnings = sorted(earnings.items(), key=lambda x: x[1], reverse=True)

        # # Iterate through the sorted list to find the ranking of the current id
        # for id_, earnings_value in sorted_earnings:
        #     self.logger.info(f"id: {id_}, earnings: {earnings_value}")


        for rank, (id_, earnings_value) in enumerate(sorted_earnings, 1):
            if id_ == self.id:
                return rank

        # If not found (theoretically impossible), return -1
        return -1


    def update_market(self, b2e_result: Dict):
    
        self.current_funds = b2e_result["current_funds"][self.id]
        self.logger.info(f"{self.id} 's current fund is: {self.current_funds}")
        self.logger.info(f"{self.id} 's b2e earning is {b2e_result['earnings'][self.id]}")
        self.current_funds_history.append(self.current_funds)
        
        self.b2e_revenue_history.append(b2e_result["earnings"][self.id])
        self.b2e_rate_history.append(b2e_result["earnings"][self.id] * 1.0 / self.current_funds)
        
        self.revenue_history.append(self.b2e_revenue_history[-1] * self.tax_rate)
        
        # # Deduct the revenue from the only volunteer who won't exit
        net_revenue_rate = (self.current_funds - self.initial_funds) * 1.0 / self.current_funds * self.tax_rate
        
        # self.logger.info(f"{self.id} : {net_revenue_rate},{self.current_funds - self.initial_funds}/{self.current_funds}")
        net_revenue = self.b2e_revenue_history[-1] * net_revenue_rate
        
        self.net_revenue_history.append(net_revenue)
        
        self.tax_rate_history.append(self.tax_rate)
        
        
        self.logger.info(f"{self.id} 's principal earning is {b2e_result['earnings'][self.id] * (self.current_funds - self.initial_funds) * 1.0 / self.current_funds}")
        self.logger.info(f"{self.id} 's MER earning is {net_revenue}")
        self.logger.info(f"{self.id} earn rank is {self.get_earnings_rank(b2e_result)}")
        

   
    def make_decision(self, market_data: Dict) -> Dict:
        """
        Make decisions based on market data, primarily optimizing tax rate
        """
        self.iteration=self.iteration+1

        b2e_rates = list(market_data['b2e_rates'].values())
        b2e_earnings = list(market_data['b2e_earnings'].values())
        num_users = market_data['num_users']
        total_investment = market_data['total_investment']
        current_funds = market_data['current_funds']
        transaction_data = market_data['transaction_data']
        # self.logger.info(transaction_data)
        
        # self.logger.info(f"======================{self.id}==========================")
        # self.logger.info(b2e_rates)
        # self.logger.info(b2e_earnings)
        # self.logger.info(num_users)
        # self.logger.info(total_investment)
        # self.logger.info(current_funds)
        # self.logger.info(f"======================{self.id}==========================")
        
        if(self.optimizer_type == "manage"):
            self.new_tax_rate = self.tax_optimizer.optimize(
                iteration=len(self.tax_rate_history),
                last_b2e_rates=market_data['b2e_rates'],
                last_b2e_earnings=market_data['b2e_earnings'],
                participation_rate1= len(self.users)/len(current_funds),
                total_investment=total_investment,
                current_funds=current_funds,
                current_earn = self.net_revenue_history[-1],
                transaction_data = transaction_data
            )
        elif(self.optimizer_type == "sym"):
            self.new_tax_rate = self.tax_rate
        self.tax_rate = self.new_tax_rate

        # self.logger.info(f"{self.id} net_revenue is {self.net_revenue_history[-1]}")
        self.logger.info(f"{self.id} tax_rate is {self.tax_rate}")
        return {
            'tax_rate': self.tax_rate
        }
        
    def update_decision(self, acc: Dict):
        
        self.current_funds = self.initial_funds + self.total_user_funds
        
        # Whether to accumulate revenue
        if(acc["acc"]):
            self.initial_funds += self.revenue_history[-1]
            self.current_funds += self.revenue_history[-1]
        
        
    def add_user(self, user_id: int, user_funds: float):
        self.users.add(user_id)
        self.total_user_funds += user_funds

    def remove_user(self, user_id: int, user_funds: float):
        self.users.remove(user_id)
        self.total_user_funds -= user_funds
    
    def get_state(self) -> Dict:
        """
        Return the current state of the LiquidityPool
        """
        return {
            'id': self.id,
            'current_funds': self.current_funds,
            'b2e_revenue': self.b2e_revenue_history[-1],
            'revenue': self.revenue_history[-1],
            'net_revenue': self.net_revenue_history[-1],
            'tax_rate': self.tax_rate,
            # 'historical_max_successful_rate': self.historical_max_successful_rate_history[-1],
            'users': list(self.users),
            'total_user_funds': self.total_user_funds
        }
    def calculate_user_share(self, total_earnings: float, user_funds: float) -> float:
        user_share = total_earnings * (1 - self.tax_rate) * (user_funds / self.total_user_funds)
        return user_share
    
    def calculate_expected_revenue(self, total_earnings: float) -> float:
        """
        Calculate liquiditypool expected revenue
        """
        return self.tax_rate * total_earnings

    def get_user_earnings_rate(self) -> float:
        """
        Calculate the expected earnings rate for users in this LiquidityPool
        """
        return (1 - self.tax_rate)  * self.b2e_rate_history[-1]