import numpy as np
from typing import List, Dict
from scipy.optimize import minimize_scalar,nnls
from typing import List, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

class ImprovedTaxRateOptimizer:
    def __init__(self, config: Dict):
    # def __init__(self, initial_delta: float, learning_rate: float = 0.01, memory_size: int = 5):
        self.id = config["params"]["id"]
        self.initial_funds = config["params"]["initial_funds"]
        self.delta = config["params"]["initial_tax_rate"]
        self.learning_rate = config["params"]["learning_rate"]
        self.memory_size = config["params"]["memory_size"]
        self.revenue_history: List[float] = []
        self.participation_history: List[float] = []
        self.delta_history: List[float] = [config["params"]["initial_tax_rate"]]
        self.last_participation_rate = 0
        
        self.earnings_history: List[float] = []
        self.earnings_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        self.min_data_points = config["params"]["min_data_points"]  # Minimum number of data points required to build model
        
        # Added: b2e investment-earnings model
        self.b2e_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.b2e_data_X = []
        self.b2e_data_y = []
        
        # Added: delta-liquiditypool investment model
        self.max_delta = config["params"]["max_tax_rate"]
        self.min_delta = config["params"]["min_tax_rate"]
        self.delta_investment_model = None
        self.delta_data = []
        self.investment_data = []
        self.history_transaction_data=[]

        # Initialize regression models for fee and amount
        self.scaler = StandardScaler()
        self.b2e_model_fee = LinearRegression()
        self.b2e_model_amount = LinearRegression()
                        

    # # Removed: Changed to use regression model to predict current round's cross-shard transaction info, and estimate current round's earnings based on this prediction
    # def update_b2e_model(self, investments: List[float], earnings: List[float], transaction_data: List[Tuple[int, int, str, str]]):
    #     # print(f"======================b2e_model==========================")
    #     # print(investments)
    #     # print(earnings)
    #     # print(f"======================b2e_model==========================")
        
    #     if len(self.b2e_data_X) > 1000:  # Limit number of data points
    #         self.b2e_data_X = self.b2e_data_X[-1000:]
    #         self.b2e_data_y = self.b2e_data_y[-1000:]
        
    #     if len(self.b2e_data_X) > 1:  # Ensure enough data to train model
    #         self.b2e_model.fit(np.array(self.b2e_data_X).reshape(-1, 1), self.b2e_data_y)
    #     else:
    #         print("Not enough data to train B2E model")   

    # def predict_b2e_earnings(self, investment: float) -> float:
    #     if len(self.b2e_data_X) < 10:  # If insufficient data, return an estimate based on simple proportion
    #         return investment * (sum(self.b2e_data_y) / sum(self.b2e_data_X)) if sum(self.b2e_data_X) > 0 else investment
    #     return self.b2e_model.predict([[investment]])[0]


    ## Modified: Regress current epoch transaction information
    def update_b2e_model(self, transaction_data: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, str, str]]:
        """
        Update model based on each round's transaction data and predict all transaction data for current round.
        transaction_data: Current round's transaction data, containing each transaction's (fee, amount, sender, receiver)
        """

        # Update historical transaction data, ensure at most 10 rounds of data are kept
        self.history_transaction_data.append(transaction_data)
        if len(self.history_transaction_data) > 10:
            self.history_transaction_data.pop(0)  # Remove oldest round of transaction data

        # Get current rounds of data being used
        history_data_len = len(self.history_transaction_data)
        #print(f"Using {history_data_len} rounds of data for training.")

        # Construct training data
        X = []  # Feature data
        y_fee = []  # fee labels
        y_amount = []  # amount labels

        # Use most recent data for training
        for epoch_transactions in self.history_transaction_data:
            for fee, amount, sender, receiver in epoch_transactions:
                # Extract features: we can add more features based on actual needs
                X.append([fee, amount])  # Features: only using fee and amount here
                y_fee.append(fee)
                y_amount.append(amount)

        # Debug output: view training data
        # print(f"Training data (X): {X}")
        # print(f"Training data (y_fee): {y_fee}")
        # print(f"Training data (y_amount): {y_amount}")

        # Standardize input features
        X_scaled = self.scaler.fit_transform(X)  # Standardize features of all historical data

        # Train models
        self.b2e_model_fee.fit(X_scaled, y_fee)  # Train fee prediction model
        self.b2e_model_amount.fit(X_scaled, y_amount)  # Train amount prediction model

        # print("Model updated with historical data.")

        # Use model to predict all transaction data for current round
        predicted_transaction_data = []
        total_transaction = 0

        for epoch_transactions in self.history_transaction_data:
            # Predict each transaction
            for fee, amount, sender, receiver in epoch_transactions:
                # Predict current transaction data
                feature = self.scaler.transform([[fee, amount]])  # Standardize current data
                predicted_fee = self.b2e_model_fee.predict(feature)[0]  # Predict fee
                predicted_amount = self.b2e_model_amount.predict(feature)[0]  # Predict amount
                predicted_transaction_data.append({
                    'predicted_fee': predicted_fee,
                    'predicted_amount': predicted_amount,
                    'sender': sender,
                    'receiver': receiver
                })
                total_transaction += predicted_amount

        # Debug output: view predicted transaction data
        # print(f"Predicted total cross-shard transaction amount: {total_transaction}")

        return predicted_transaction_data  # Return predicted transaction data
    
    ## Greedy algorithm to predict earnings
    def predict_b2e_earnings(self, predicted_transaction_data: List[dict], predicted_investments: float) -> float:
        """
        Use greedy algorithm to calculate predicted earnings based on predicted transaction data and investment amount.
        - Sort by transaction return rate (predicted_fee / predicted_amount) from high to low, execute one by one
        - Ensure each cross-shard transaction amount is less than corresponding investment amount.

        Args:
            predicted_transaction_data (List[dict]): Predicted transaction data, each transaction contains 'predicted_fee' and 'predicted_amount'.
            predicted_investments (float): Total investment amount.

        Returns:
            predicted_earnings (float): Calculated total predicted earnings.
        """
        # Sort transaction data by predicted_fee / predicted_amount from high to low
        predicted_transaction_data_sorted = sorted(predicted_transaction_data, 
                                                key=lambda x: x['predicted_fee'] / x['predicted_amount'], 
                                                reverse=True)

        total_earnings = 0.0
        for i in range(len(predicted_transaction_data_sorted)):
            transaction = predicted_transaction_data_sorted[i]

            predicted_fee = transaction['predicted_fee']
            predicted_amount = transaction['predicted_amount']
            
            # Ensure each transaction amount is less than corresponding investment amount
            if predicted_amount <= predicted_investments:
                predicted_transaction_earnings = predicted_fee + predicted_amount
                total_earnings += predicted_transaction_earnings
                predicted_investments -= predicted_amount  # Update remaining investment amount
            # else:
            #     print(f"Transaction {i+1} exceeds investment limit, skipping it.")

        return total_earnings

    # def update_delta_investment_model(self, delta: float, total_investment: float):
    #     self.delta_data.append(delta)
    #     self.investment_data.append(total_investment)
    #     nums = 50
    #     if len(self.delta_data) > nums:  # Limit number of data points
    #         self.delta_data = self.delta_data[-nums:]
    #         self.investment_data = self.investment_data[-nums:]
        
    #     if len(self.delta_data) > 2:  # Ensure enough data points for fitting
    #         X = np.array(self.delta_data).reshape(-1, 1)
    #         y = np.array(self.investment_data)
            
    #         # Data normalization
    #         scaler_X = StandardScaler()
    #         scaler_y = StandardScaler()
    #         X_scaled = scaler_X.fit_transform(X)
    #         y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    #         # Define step function and polynomial regression
    #         def step_function(x, a, b, c):
    #             return a * np.heaviside(x - b, 1) + c

    #         best_model = None
    #         best_score = -np.inf

    #         # Try step function
    #         try:
    #             popt, _ = curve_fit(step_function, X_scaled.ravel(), y_scaled)
    #             score = r2_score(y_scaled, step_function(X_scaled.ravel(), *popt))
    #             best_model = ("Step function", lambda x: step_function(x, *popt), score)
    #             best_score = score
    #         except RuntimeError:
    #             print("")
    #             # print("Unable to fit step function")

    #         # Try polynomial regression (2nd and 3rd degree)
    #         for degree in [2, 3]:
    #             poly = PolynomialFeatures(degree=degree)
    #             X_poly = poly.fit_transform(X_scaled)
    #             poly_reg = LinearRegression()
    #             poly_reg.fit(X_poly, y_scaled)
    #             score = r2_score(y_scaled, poly_reg.predict(X_poly))
    #             if score > best_score:
    #                 best_score = score
    #                 best_model = (f"{degree}-degree polynomial regression", 
    #                               lambda x: poly_reg.predict(poly.transform(x.reshape(-1, 1))), 
    #                               score)

    #         if best_model:
    #             name, func, score = best_model
    #             # print(f"Best model: {name}, R² score: {score:.4f}")
    #             self.delta_investment_model = lambda x: scaler_y.inverse_transform(
    #                 func(scaler_X.transform([[x]])).reshape(-1, 1)
    #             ).ravel()[0]
    #         else:
    #             # print("All model fitting failed, using simple linear regression")
    #             reg = LinearRegression().fit(X, y)
    #             self.delta_investment_model = lambda x: reg.predict([[x]])[0]
    #     else:
    #         # print("Insufficient data points, using average value")
    #         self.delta_investment_model = lambda x: np.mean(self.investment_data)
        
    def update_delta_investment_model(self, delta: float, total_investment: float):
        self.delta_data.append(delta)
        self.investment_data.append(total_investment)
        nums = 50  # Limit number of data points
        if len(self.delta_data) > nums:
            self.delta_data = self.delta_data[-nums:]
            self.investment_data = self.investment_data[-nums:]
        
        # Number of data points
        data_length = len(self.delta_data)
        
        if data_length >= 2:
            # Check and filter non-numeric data
            valid_data = [
                (d, i) for d, i in zip(self.delta_data, self.investment_data)
                if isinstance(d, (int, float)) and isinstance(i, (int, float))
            ]
            
            if not valid_data:
                # If no valid data points, use average investment for prediction
                avg_investment = max(0, np.mean([
                    i for i in self.investment_data if isinstance(i, (int, float))
                ]))
                self.delta_investment_model = lambda x: avg_investment
                # print(f"LiquidityPool {self.id}: No valid data, using average investment {avg_investment:.4f} for prediction")
                return
            
            # Extract valid delta and investment data
            X = np.array([d for d, _ in valid_data], dtype=float).reshape(-1, 1)
            y = np.array([i for _, i in valid_data], dtype=float)
            
            # Build design matrix, add constant term
            X_design = np.hstack([X, np.ones_like(X)])
            
            # Use non-negative least squares for linear fitting
            coeffs_nnls, _ = nnls(X_design, y)
            y_pred_nnls = X_design @ coeffs_nnls
            score_nnls = r2_score(y, y_pred_nnls)
            best_model = ("Non-negative linear regression", lambda x: max(0, coeffs_nnls[0]*x + coeffs_nnls[1]), score_nnls)
            best_score = score_nnls
            
            # When data is sufficient, try polynomial regression models
            if data_length >= 10:
                for degree in [2, 3]:
                    poly = PolynomialFeatures(degree=degree)
                    X_poly = poly.fit_transform(X)
                    model = LinearRegression()
                    model.fit(X_poly, y)
                    y_pred = model.predict(X_poly)
                    score = r2_score(y, y_pred)
                    
                    # If new model has higher score, update best model
                    if score > best_score:
                        best_model = (
                            f"{degree}-degree polynomial regression",
                            lambda x: max(0, model.predict(poly.transform([[x]]))[0]),
                            score
                        )
                        best_score = score
            
            # Select best model
            model_name, model_func, model_score = best_model
            self.delta_investment_model = model_func
            # print(f"LiquidityPool {self.id}: Using model '{model_name}' for fitting, R² score is {model_score:.4f}")
        else:
            # Insufficient data points, use average value
            avg_investment = max(0, np.mean(self.investment_data))
            self.delta_investment_model = lambda x: avg_investment
            # print(f"LiquidityPool {self.id}: Insufficient data, using average investment {avg_investment:.4f} for prediction")


    def predict_investment(self, delta: float) -> float:
        if self.delta_investment_model is None:
            return max(0, np.mean(self.investment_data)) if self.investment_data else 0 # Ensure non-negative
        predicted_investment = self.delta_investment_model(delta)
        # print(f"LiquidityPool {self.id}: For tax rate {delta}, predicted investment value is: {predicted_investment}")
        return predicted_investment

    def optimize(self, iteration: int, last_b2e_rates: List[float], last_b2e_earnings: List[float], participation_rate1: float, total_investment: float, current_funds: List[float], current_earn: float, transaction_data: List[Tuple[int, int, str, str]]) -> float:
        
        # print(f"LiquidityPool {self.id}: Earnings: {current_earn}, Previous round participation rate: {participation_rate1}, Total invested funds: {total_investment}, of which {self.id} accounts for {current_funds[self.id]}")
        # Add earnings threshold detection
        if iteration > 1 and participation_rate1 < 0.1:
            # If participation rate is too low, proactively reduce tax rate
            self.delta = max(self.min_delta, self.delta * 0.9)  # Reduce tax rate to 90%, ensuring it doesn't go below minimum
            self.delta_history.append(self.delta)
            # print(f"LiquidityPool {self.id}: Due to low participation rate, reducing tax rate to {self.delta}")
            return self.delta
        # Update models
        # self.update_b2e_model(current_funds, last_b2e_earnings)
        self.update_delta_investment_model(self.delta, current_funds[self.id])
        
        # Update revenue history
        current_revenue = current_earn
        self.revenue_history.append(current_revenue + 1e-5)
        
        # Update participation rate history
        self.participation_history.append(participation_rate1)
    
        # Calculate short-term revenue trend
        short_term_trend = self.calculate_short_term_trend()

        ## Added: Predict current epoch transaction information
        predicted_transaction_data = self.update_b2e_model(transaction_data)

        def objective(delta: float) -> float:
            # Predict investments    
            predicted_investment = self.predict_investment(delta)
            predicted_earnings = self.predict_b2e_earnings(predicted_transaction_data, predicted_investment)
            
            # Remove earnings from the only user
            predicted_earnings = (predicted_earnings - self.initial_funds) * 1.0 / (predicted_earnings + 1e-7) * delta 
            
            # print(f"Current tax rate: {delta}, expected earnings: {predicted_earnings}")

            liquiditypool_rate = (1 - delta) * predicted_earnings / predicted_investment if predicted_investment > 0 else 0
            participation_rate = participation_rate1
            expected_revenue = delta * predicted_earnings * 1e-11  

            return expected_revenue
            
            # Add historical trend factor
            trend_factor = self.calculate_trend_factor()
            
            # Add volatility penalty
            volatility_penalty = self.calculate_volatility_penalty(delta)
            
            # Long-term stability reward
            stability_reward = self.calculate_stability_reward(delta, participation_rate)
            
            participation_factor = (self.last_participation_rate - participation_rate) * 1e4  # Add participation rate factor

            self.last_participation_rate = participation_rate
                        
            # Adjust objective function based on short-term trend
            trend_adjustment = 1 + short_term_trend * 3  # Increase influence of short-term trend
            # Add revenue decline penalty
            revenue_decline_penalty = 0
            if len(self.revenue_history) > 1 and self.revenue_history[-1] < self.revenue_history[-2]:
                revenue_decline_penalty = (self.revenue_history[-2] - self.revenue_history[-1]) / self.revenue_history[-2] * 1e5
            # Calculate revenue volatility
            if len(self.revenue_history) >= 5:
                revenue_volatility = np.std(self.revenue_history[-5:]) / np.mean(self.revenue_history[-5:])
            else:
                revenue_volatility = 0
            
            revenue_volatility_penalty = revenue_volatility * 1e5  # Increase penalty for revenue volatility
            
            return -(expected_revenue * trend_adjustment + 
                     participation_rate * expected_revenue * 2 + 
                     stability_reward - 
                     volatility_penalty * 1.5 - 
                     revenue_decline_penalty - 
                     revenue_volatility_penalty)
            # Stable code
            # trend_adjustment = 1 + short_term_trend

            # return -(expected_revenue * trend_adjustment + 
                     # participation_rate * expected_revenue * 0.5 + 
                     # stability_reward * 10 -  # Increase weight of stability reward
                     # volatility_penalty * 0.1)  # Reduce impact of volatility penalty


        result = minimize_scalar(objective, bounds=(self.min_delta, self.max_delta), method='bounded')
        new_delta = result.x
        
        # Dynamically adjust learning rate
        revenue_volatility = np.std(self.revenue_history[-5:]) / np.mean(self.revenue_history[-5:]) if len(self.revenue_history) >= 5 else 0
        adaptive_learning_rate = self.learning_rate * (1 + revenue_volatility * 2 )  # Increase learning rate based on revenue volatility

        # Apply adaptive learning rate
        self.delta = max(min(self.delta + adaptive_learning_rate * (new_delta - self.delta), self.max_delta), self.min_delta)
        # Update delta history
        
        # if self.detect_stagnation():
        #     # If stagnation detected, force a larger adjustment
        #     self.delta = max(min(self.delta * (1 + np.random.uniform(-0.2, 0.2)), self.max_delta), self.min_delta)
    
        self.delta = max(min(self.delta, self.max_delta), self.min_delta)
        
        self.delta_history.append(self.delta)
        
        return self.delta
    
    def detect_stagnation(self) -> bool:
        if len(self.delta_history) < 20:
            return False
        recent_changes = np.diff(self.delta_history[-20:])
        return np.all(np.abs(recent_changes) < 1e-5)  # If tax rate changes in last 20 iterations are all very small, consider it stagnant

    def calculate_short_term_trend(self) -> float:
        if len(self.revenue_history) < 5:
            return 0
        recent_trend = (self.revenue_history[-1] - np.mean(self.revenue_history[-5:-1])) / np.mean(self.revenue_history[-5:-1])
        return np.clip(recent_trend * 5, -1, 1)  # Amplify trend influence and limit to [-1, 1] range
        # return np.clip(recent_trend, -0.1, 0.1)  # Stable code        
   
    def calculate_trend_factor(self) -> float:
        if len(self.revenue_history) < 2:
            return 0
        recent_trend = (self.revenue_history[-1] - self.revenue_history[-2]) / max(self.revenue_history[-2], 1)
        return recent_trend * 500  # Amplify trend influence

    def calculate_volatility_penalty(self, proposed_delta: float) -> float:
        if len(self.delta_history) < 2:
            return 0
        recent_volatility = abs(proposed_delta - self.delta_history[-1])
        return recent_volatility * 2000  # Amplify volatility penalty

    def calculate_stability_reward(self, proposed_delta: float, participation_rate: float) -> float:
        if len(self.participation_history) < self.memory_size:
            return 0
        avg_participation = np.mean(self.participation_history[-self.memory_size:])
        stability_score = 1 - abs(participation_rate - avg_participation)
        return stability_score * 10  # Amplify stability reward
        # Stable code
        # delta_stability = 1 - abs(proposed_delta - self.delta) / self.delta
        # return (stability_score * 100 + delta_stability * 1000)  # Significantly increase stability reward

    def get_statistics(self) -> Tuple[List[float], List[float], List[float]]:
        return self.delta_history, self.revenue_history, self.participation_history