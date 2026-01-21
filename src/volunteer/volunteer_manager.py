import random
from typing import List, Dict
from .volunteer import Volunteer

class VolunteerManager:
    def __init__(self, config: Dict):
        self.config = config
        self.volunteers = self._initialize_volunteers()

    def _initialize_volunteers(self) -> List[Volunteer]:
        volunteers = []
        with open(self.config['brokerBalancePath'], 'r') as f:
            balances = [int(line.strip()) for line in f]
        
        risk_min, risk_max = self.config['volunteerRiskToleranceRange']
        adding = self.config['Balance_accumulation']
        
        for i, balance in enumerate(balances):
            risk_tolerance = random.uniform(risk_min, risk_max)
            volunteers.append(Volunteer(i, balance, risk_tolerance, adding))
        
        return volunteers

    def make_decisions(self, market_data: Dict, brokerhubs: List[Dict]) -> List[Dict]:
        """
        为所有Volunteer做出决策
        """
        decisions = {}
        for volunteer in self.volunteers:
            decision = volunteer.make_decision(market_data, brokerhubs)
            decisions.update({
                volunteer.id: decision
            })
        return decisions

    def update_market(self, b2e_results: List[Dict], brokerhubs: List[Dict]):
        """
        根据市场结果更新所有Volunteer的状态
        """
        for volunteer in self.volunteers:
            volunteer.update_market(b2e_results, brokerhubs)
            
    def update_decisions(self, vol_decisions: List[Dict], brokerhubs: List[Dict]):
        """
        根据市场结果更新所有Volunteer的状态
        """
        for volunteer in self.volunteers:
            volunteer.update_decision(vol_decisions[volunteer.id], brokerhubs)
            
    def get_states(self) -> List[Dict]:
        """
        获取所有Volunteer的当前状态
        """
        return [volunteer.get_state() for volunteer in self.volunteers]