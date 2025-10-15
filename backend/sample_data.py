"""
Enhanced sample data with location information for Rake Formation System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_enhanced_dataset():
    """Generate enhanced dataset with location information"""
    
    # Base locations
    origins = ['Bokaro Steel Plant', 'Rourkela Steel Plant', 'Bhilai Steel Plant', 'Durgapur Steel Plant']
    destinations = [
        'Tata Steel Jamshedpur', 'Jindal Steel Raigarh', 'Essar Steel Hazira', 
        'SAIL Burnpur', 'RINL Visakhapatnam', 'JSW Steel Dolvi', 'Bhushan Steel Meramandali'
    ]
    
    # States for broader geographical context
    states = ['Jharkhand', 'Odisha', 'Chhattisgarh', 'West Bengal', 'Gujarat', 'Andhra Pradesh', 'Maharashtra']
    
    # Generate sample data
    n_records = 1000
    data = []
    
    for i in range(n_records):
        plan_id = f"PLAN_{2024}_{i+1:04d}"
        
        # Random dates in next 30 days
        plan_date = datetime.now() + timedelta(days=random.randint(1, 30))
        
        # Customer information
        customer_name = f"Customer_{random.randint(1, 50)}"
        
        # Location information
        origin = random.choice(origins)
        destination = random.choice(destinations)
        origin_state = random.choice(states)
        destination_state = random.choice([s for s in states if s != origin_state])
        
        # Transportation details
        distance_km = random.randint(200, 1500)
        planned_qty_t = random.randint(50, 500)
        min_rake_tonnage = random.randint(100, 300)
        
        # Cost factors
        terminal_cost = random.randint(5000, 20000)
        expected_demurrage = random.randint(1000, 8000)
        priority_score = round(random.uniform(0.1, 1.0), 2)
        
        # Operational constraints
        rake_available = random.choice([0, 1])
        siding_slots = random.randint(1, 5) if rake_available else 0
        inventory_t = random.randint(1000, 10000)
        production_forecast_t = random.randint(500, 5000)
        
        # Yard information
        yard_id = f"YARD_{random.randint(1, 10)}"
        
        data.append({
            'plan_id': plan_id,
            'plan_date': plan_date.strftime('%d/%m/%Y'),
            'customer_name': customer_name,
            'origin_plant': origin,
            'origin_state': origin_state,
            'destination': destination,
            'destination_state': destination_state,
            'distance_km': distance_km,
            'planned_qty_t': planned_qty_t,
            'min_rake_tonnage': min_rake_tonnage,
            'terminal_cost': terminal_cost,
            'expected_demurrage': expected_demurrage,
            'priority_score': priority_score,
            'rake_available': rake_available,
            'siding_slots': siding_slots,
            'inventory_t': inventory_t,
            'production_forecast_t': production_forecast_t,
            'yard_id': yard_id,
            'product_type': random.choice(['Coal', 'Iron Ore', 'Limestone', 'Finished Steel']),
            'wagon_type': random.choice(['BOXN', 'BCN', 'BTPN', 'BRN']),
            'route_via': random.choice(['Direct', 'Via Tatanagar', 'Via Bilaspur', 'Via Nagpur']),
            'lead_time_days': random.randint(2, 10)
        })
    
    return pd.DataFrame(data)

def save_enhanced_dataset():
    """Generate and save enhanced dataset"""
    df = generate_enhanced_dataset()
    df.to_csv('./enhanced_bokaro_customers.csv', index=False)
    print(f"✅ Enhanced dataset generated with {len(df)} records")
    print("📊 Dataset columns:", df.columns.tolist())
    return df

if __name__ == "__main__":
    save_enhanced_dataset()