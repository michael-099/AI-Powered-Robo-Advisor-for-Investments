# Synthetic robo-advisor dataset generator
# Paste into a Python cell and run. Saves "synthetic_robo_data.csv" to working dir.
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

np.random.seed(42)

def trunc_normal(mean=0, sd=1, low=-np.inf, high=np.inf, size=1):
    a, b = (low - mean) / sd, (high - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

def generate_synthetic_robo_data(n=5000):
    # 1. Basic numeric features (age, income, net_worth) with realistic variance

    # Age: bimodal distribution (young adults + mid-career)
    age = np.concatenate([
        np.random.normal(28, 5, size=int(n*0.35)),
        np.random.normal(42, 8, size=int(n*0.45)),
        np.random.normal(60, 6, size=int(n*0.20)),
    ]).astype(int)
    age = np.clip(age, 18, 75)
    
    # Annual income (log-normal — real world distribution)
    annual_income = np.random.lognormal(mean=10.5, sigma=0.55, size=n).astype(int)
    annual_income = np.clip(annual_income, 800, 2_000_000)
    
    # Net worth (also log-normal, but more spread)
    net_worth = np.random.lognormal(mean=11.2, sigma=1.0, size=n).astype(int)
    
    # Generate a multiplicative factor between 0.05 and 4.5
    noise_factor = np.random.uniform(low=0.05, high=4.5, size=n)

    # Multiply the deterministic component by this random factor
    # **REVISED:** Additive and Multiplicative noise to Net Worth calculation
    net_worth = (np.abs(0.08 * annual_income**1.1 + np.random.normal(0, 10000, size=n)) * noise_factor).astype(int)
    net_worth = np.clip(net_worth, 0, 50_000_000) # Re-clip net worth

    
    # Add realistic correlation: richer → higher income
    annual_income.sort()
    net_worth.sort()
    

    # 2. Other numeric / financial metrics
    savings_rate = np.clip(trunc_normal(0.15, 0.08, 0, 0.8, size=n), 0, 1)  
    debt_ratio = np.clip(trunc_normal(0.25, 0.15, 0, 1, size=n), 0, 1)    
    employment_years = np.clip(age - np.random.randint(16, 25, size=n), 0, 50)
    investment_experience_years = np.clip(trunc_normal(5, 6, 0, 50, size=n).round().astype(int), 0, 50)
    time_horizon_years = np.clip(trunc_normal(10, 8, 1, 60, size=n).round().astype(int), 1, 60)
    emergency_fund_months = np.clip(trunc_normal(3, 2, 0, 36, size=n).round().astype(int), 0, 36)

    # 3. Psychometric questionnaire (5 items, 1-5 Likert)
    q = np.clip((np.random.normal(3, 1, size=(n,5))).round(), 1, 5).astype(int)

    # 4. Categorical features (controlled cardinality)
    employment_status = np.random.choice(['employee','self-employed','student','retired','unemployed'], size=n, p=[0.6,0.15,0.05,0.15,0.05])
    primary_goal = np.random.choice(['retirement','wealth_generation','education','home_purchase','income'], size=n, p=[0.35,0.35,0.10,0.10,0.10])
    liquidity_need = np.random.choice(['low','medium','high'], size=n, p=[0.6,0.3,0.1])
    resources_used = np.random.choice(['books','advisor','online','friends','none'], size=n, p=[0.25,0.2,0.35,0.1,0.1])

    # 5. Derived scores: risk_tolerance_score and risk_capacity_score
    q_weights = np.array([0.25, 0.25, 0.2, 0.15, 0.15])
    risk_tolerance = (q * q_weights).sum(axis=1) / (5 * q_weights.sum())  # normalized 0..1

    income_scaled = (np.log1p(annual_income) - np.log1p(500)) / (np.log1p(5_000_000)-np.log1p(500))
    net_worth_scaled = (np.log1p(np.clip(net_worth,0,None)+1) - 0) / (np.log1p(50_000_000)+1)
    time_horizon_scaled = (time_horizon_years - 1) / 59.0
    employment_score = np.where(employment_status=='employee', 1.0, 
                             np.where(employment_status=='self-employed', 0.8,
                             np.where(employment_status=='retired', 0.6,
                             np.where(employment_status=='student', 0.5, 0.4))))
    debt_penalty = 1 - debt_ratio
    risk_capacity = (0.4*income_scaled + 0.3*net_worth_scaled + 0.2*time_horizon_scaled + 0.1*employment_score) * debt_penalty
    risk_capacity = np.clip(risk_capacity, 0, 1)

    # 6. Combine tolerance & capacity into a score and label
    combined_score = 0.6 * risk_tolerance + 0.4 * risk_capacity 
    # **REVISED:** Increase the general noise added to the combined score
    combined_score = np.clip(combined_score + np.random.normal(0, 0.05, size=n), 0, 1)

    # Determine investor_type thresholds (tuneable)
    # **CRITICAL REVISION:** Use a fuzzy boundary based on probability
    
    # Thresholds for classification
    moderate_low = 0.35
    moderate_high = 0.65
    
    investor_type = np.full(n, 'moderate', dtype='object') # Default to moderate
    
    # Conservative assignment: combined_score near/below 0.35
    # Use a sigmoid-like probability curve for classification noise
    prob_conservative = 1 / (1 + np.exp( (combined_score - moderate_low) / 0.05 ))
    is_conservative = np.random.rand(n) < prob_conservative
    investor_type[is_conservative] = 'conservative'
    
    # Aggressive assignment: combined_score near/above 0.65
    prob_aggressive = 1 / (1 + np.exp( (moderate_high - combined_score) / 0.05 ))
    is_aggressive = np.random.rand(n) < prob_aggressive
    investor_type[is_aggressive] = 'aggressive'
    
    # 7. Recommended allocation (simple deterministic mapping + time_horizon tweak)
    def recommend_alloc(itype, th):
        # base allocations (stocks, bonds, cash, alternatives)
        if itype == 'aggressive':
            base = np.array([0.75, 0.15, 0.05, 0.05])
        elif itype == 'moderate':
            base = np.array([0.55, 0.30, 0.10, 0.05])
        else:  # conservative
            base = np.array([0.30, 0.50, 0.15, 0.05])
        
        # tweak by time horizon: longer horizon -> nudge towards stocks
        tweak = np.clip((th - 5) / 50.0, -0.2, 0.2) 
        base[0] = np.clip(base[0] + tweak, 0, 0.95)
        # renormalize
        base = np.clip(base, 0, 1)
        base /= base.sum()
        
        # **REVISED:** Add noise to the final recommended allocations to reflect real-world variance
        base += np.random.normal(0, 0.02, size=base.shape)
        base = np.clip(base, 0, 1)
        base /= base.sum()
        
        return base

    allocations = np.vstack([recommend_alloc(it, th) for it, th in zip(investor_type, time_horizon_years)])
    stocks, bonds, cash, alts = allocations[:,0], allocations[:,1], allocations[:,2], allocations[:,3]

    # 8. Build DataFrame
    df = pd.DataFrame({
        'age': age,
        'annual_income': annual_income,
        'net_worth': net_worth,
        'savings_rate': savings_rate.round(3),
        'debt_ratio': debt_ratio.round(3),
        'employment_years': employment_years,
        'investment_experience_years': investment_experience_years,
        'time_horizon_years': time_horizon_years,
        'emergency_fund_months': emergency_fund_months,
        'employment_status': employment_status,
        'primary_goal': primary_goal,
        'liquidity_need': liquidity_need,
        'resources_used': resources_used,
        'q1': q[:,0], 'q2': q[:,1], 'q3': q[:,2], 'q4': q[:,3], 'q5': q[:,4],
        'risk_tolerance': risk_tolerance.round(3),
        'risk_capacity': risk_capacity.round(3),
        'combined_score': combined_score.round(3),
        'investor_type': investor_type,
        'stocks_target': np.round(stocks,3),
        'bonds_target': np.round(bonds,3),
        'cash_target': np.round(cash,3),
        'alts_target': np.round(alts,3),
    })

    # Optional: add a small synthetic current_portfolio to be used as input
    # random small deviations from recommended allocation
    noise = np.random.normal(0, 0.03, size=allocations.shape)
    current_alloc = np.clip(allocations + noise, 0, None)
    current_alloc = current_alloc / current_alloc.sum(axis=1, keepdims=True)
    df['curr_stock'] = current_alloc[:,0].round(3)
    df['curr_bonds'] = current_alloc[:,1].round(3)
    df['curr_cash'] = current_alloc[:,2].round(3)
    df['curr_alts'] = current_alloc[:,3].round(3)

    return df

# Generate and save
df_syn = generate_synthetic_robo_data(n=5000)
df_syn.to_csv('synthetic_robo_data.csv', index=False)
print("Saved synthetic_robo_data.csv — shape:", df_syn.shape)
df_syn.head()