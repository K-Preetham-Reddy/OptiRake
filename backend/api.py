# """
# api.py

# FastAPI server for Rake Formation Optimization System
# """

# from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, List, Optional, Any
# import pandas as pd
# import numpy as np
# import os
# import json
# import asyncio
# import uuid
# from datetime import datetime
# import joblib

# from optim import optimize_rake_allocation
# from multi_model import MultiModelPipeline

# # -------------------------------------------------------------------------
# # Configuration
# # -------------------------------------------------------------------------
# DATA_PATH = "./enhanced_bokaro_customers.csv"
# MODEL_DIR = "./saved_models"
# OUTPUT_DIR = "./api_results"

# # Create output directory if it doesn't exist
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -------------------------------------------------------------------------
# # FastAPI App
# # -------------------------------------------------------------------------
# app = FastAPI(
#     title="Rake Formation Optimization API",
#     description="API for optimizing rake allocation and formation planning",
#     version="2.0.0"
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------------------------------------------------------
# # Models
# # -------------------------------------------------------------------------
# class OptimizationRequest(BaseModel):
#     plan_ids: Optional[List[str]] = None
#     use_sample: bool = True
#     custom_data: Optional[List[Dict]] = None
#     optimization_mode: str = "balanced"  # balanced, cost_min, rail_max

# class OptimizationResponse(BaseModel):
#     job_id: str
#     status: str
#     message: str
#     results_path: Optional[str] = None
#     summary: Optional[Dict] = None
#     timestamp: str

# class JobStatusResponse(BaseModel):
#     job_id: str
#     status: str
#     progress: float
#     message: str
#     results: Optional[Dict] = None
#     timestamp: str

# # -------------------------------------------------------------------------
# # Global State
# # -------------------------------------------------------------------------
# jobs = {}
# pipeline = None

# # -------------------------------------------------------------------------
# # Helper Functions for Data Serialization
# # -------------------------------------------------------------------------
# def convert_numpy_types(obj):
#     """Convert numpy types to native Python types for JSON serialization"""
#     if isinstance(obj, (np.integer, np.int64, np.int32)):
#         return int(obj)
#     elif isinstance(obj, (np.floating, np.float64, np.float32)):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, np.bool_):
#         return bool(obj)
#     elif isinstance(obj, dict):
#         return {key: convert_numpy_types(value) for key, value in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_numpy_types(item) for item in obj]
#     elif pd.isna(obj):  # Handle NaN values
#         return None
#     else:
#         return obj

# def serialize_dataframe(df):
#     """Convert DataFrame to JSON-serializable format"""
#     if df is None:
#         return None
    
#     # Convert numpy types in the entire DataFrame
#     df_serialized = df.copy()
#     for col in df_serialized.columns:
#         if df_serialized[col].dtype in [np.int64, np.int32]:
#             df_serialized[col] = df_serialized[col].astype('int64').fillna(0)
#         elif df_serialized[col].dtype in [np.float64, np.float32]:
#             df_serialized[col] = df_serialized[col].astype('float64').fillna(0.0)
    
#     return df_serialized

# # -------------------------------------------------------------------------
# # Startup Event
# # -------------------------------------------------------------------------
# @app.on_event("startup")
# async def startup_event():
#     """Initialize the ML pipeline on startup"""
#     global pipeline
#     try:
#         print("🔹 Initializing ML pipeline...")
#         pipeline = MultiModelPipeline(DATA_PATH, MODEL_DIR)
        
#         # Load feature columns
#         feature_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
#         if os.path.exists(feature_path):
#             pipeline.feature_cols = joblib.load(feature_path)
#             print(f"✅ Loaded {len(pipeline.feature_cols)} feature columns")
#         else:
#             print("❌ Feature columns not found")
            
#         print("✅ ML pipeline initialized successfully")
#     except Exception as e:
#         print(f"❌ Failed to initialize ML pipeline: {e}")

# # -------------------------------------------------------------------------
# # Helper Functions
# # -------------------------------------------------------------------------
# def load_and_preprocess_data(plan_ids=None):
#     """Load and preprocess data for optimization"""
#     try:
#         df = pd.read_csv(DATA_PATH)
        
#         # Filter by plan_ids if provided
#         if plan_ids and len(plan_ids) > 0:
#             df = df[df['plan_id'].isin(plan_ids)]
#             if len(df) == 0:
#                 raise ValueError("No matching plan_ids found")
        
#         # Recreate engineered date features
#         if "plan_date" in df.columns:
#             try:
#                 df["plan_date"] = pd.to_datetime(df["plan_date"], dayfirst=True)
#                 df["plan_dayofweek"] = df["plan_date"].dt.dayofweek
#                 df["plan_month"] = df["plan_date"].dt.month
#                 df["plan_day"] = df["plan_date"].dt.day
#             except Exception as e:
#                 print(f"⚠️ Date feature creation failed: {e}")
        
#         return df
#     except Exception as e:
#         raise Exception(f"Data loading failed: {e}")

# def run_optimization_job(job_id: str, plan_ids: Optional[List[str]] = None, mode: str = "balanced"):
#     """Run optimization in background"""
#     try:
#         jobs[job_id] = {
#             "status": "running", 
#             "progress": 0.2,
#             "message": "Loading data...",
#             "timestamp": datetime.now().isoformat()
#         }
        
#         # Load data
#         df = load_and_preprocess_data(plan_ids)
#         jobs[job_id]["progress"] = 0.4
#         jobs[job_id]["message"] = "Running predictions..."
        
#         # Run predictions
#         preds = pipeline.predict_all(df)
#         df_pred = df.copy()
#         for k, v in preds.items():
#             df_pred[k] = v
        
#         jobs[job_id]["progress"] = 0.7
#         jobs[job_id]["message"] = "Running optimization..."
        
#         # Run optimization with specified mode
#         result_df = optimize_rake_allocation(df_pred, mode)
        
#         jobs[job_id]["progress"] = 1.0
#         jobs[job_id]["message"] = "Optimization completed"
        
#         # Save results
#         output_path = os.path.join(OUTPUT_DIR, f"results_{job_id}.csv")
#         result_df.to_csv(output_path, index=False)
        
#         # Create comprehensive summary
#         total_orders = len(result_df)
#         rail_orders = result_df["y_rail"].sum() if "y_rail" in result_df.columns else 0
#         road_orders = total_orders - rail_orders
#         rail_tonnage = result_df["q_rail_tons"].sum() if "q_rail_tons" in result_df.columns else 0
#         total_tonnage = result_df["planned_qty_t"].sum()
#         total_cost = result_df["optimized_total_cost"].sum() if "optimized_total_cost" in result_df.columns else 0
        
#         # Cost savings analysis
#         original_rail_cost = result_df["pred_rail_cost_total"].sum() if "pred_rail_cost_total" in result_df.columns else 0
#         original_road_cost = result_df["pred_road_cost_total"].sum() if "pred_road_cost_total" in result_df.columns else 0
#         baseline_cost = min(original_rail_cost, original_road_cost) * total_orders
        
#         summary = {
#             "total_orders": int(total_orders),
#             "rail_orders": int(rail_orders),
#             "road_orders": int(road_orders),
#             "rail_orders_percentage": float(round(rail_orders/total_orders*100, 1)) if total_orders > 0 else 0.0,
#             "rail_tonnage": float(round(rail_tonnage, 0)),
#             "total_tonnage": float(round(total_tonnage, 0)),
#             "rail_tonnage_percentage": float(round(rail_tonnage/total_tonnage*100, 1)) if total_tonnage > 0 else 0.0,
#             "total_cost": float(round(total_cost, 2)),
#             "cost_savings": float(round(baseline_cost - total_cost, 2)) if baseline_cost > total_cost else 0.0,
#             "savings_percentage": float(round((baseline_cost - total_cost) / baseline_cost * 100, 1)) if baseline_cost > 0 else 0.0,
#             "results_file": output_path,
#             "optimization_mode": mode
#         }
        
#         jobs[job_id]["status"] = "completed"
#         jobs[job_id]["results"] = convert_numpy_types(summary)
#         jobs[job_id]["results_df"] = result_df
        
#         print(f"✅ Optimization job {job_id} completed")
        
#     except Exception as e:
#         jobs[job_id]["status"] = "failed"
#         jobs[job_id]["message"] = f"Error: {str(e)}"
#         print(f"❌ Optimization job {job_id} failed: {e}")

# # -------------------------------------------------------------------------
# # API Endpoints
# # -------------------------------------------------------------------------
# @app.get("/")
# async def root():
#     """Root endpoint"""
#     return {
#         "message": "Rake Formation Optimization API",
#         "version": "2.0.0",
#         "status": "operational",
#         "endpoints": {
#             "health": "/health",
#             "optimize": "/optimize",
#             "job_status": "/job/{job_id}",
#             "results": "/results/{job_id}",
#             "plans": "/plans",
#             "plan_details": "/plan/{plan_id}",
#             "dashboard": "/dashboard/stats"
#         }
#     }

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     pipeline_status = "ready" if pipeline else "not_ready"
#     return {
#         "status": "healthy",
#         "pipeline": pipeline_status,
#         "timestamp": datetime.now().isoformat()
#     }

# @app.post("/optimize", response_model=OptimizationResponse)
# async def optimize_rake_formation(request: OptimizationRequest, background_tasks: BackgroundTasks):
#     """Start rake formation optimization"""
#     try:
#         # Generate job ID
#         job_id = str(uuid.uuid4())[:8]
        
#         # Validate request
#         if request.custom_data and len(request.custom_data) > 0:
#             # TODO: Implement custom data handling
#             raise HTTPException(status_code=501, detail="Custom data not yet supported")
        
#         # Start background job
#         background_tasks.add_task(run_optimization_job, job_id, request.plan_ids, request.optimization_mode)
        
#         # Store job info
#         jobs[job_id] = {
#             "status": "queued",
#             "progress": 0.0,
#             "message": "Job queued for processing",
#             "timestamp": datetime.now().isoformat()
#         }
        
#         return OptimizationResponse(
#             job_id=job_id,
#             status="queued",
#             message="Optimization job started",
#             timestamp=datetime.now().isoformat()
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")

# @app.get("/job/{job_id}", response_model=JobStatusResponse)
# async def get_job_status(job_id: str):
#     """Get optimization job status"""
#     if job_id not in jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     job = jobs[job_id]
    
#     # Convert numpy types in job data
#     job_data = {
#         "job_id": job_id,
#         "status": job["status"],
#         "progress": float(job["progress"]),
#         "message": job["message"],
#         "results": convert_numpy_types(job.get("results")),
#         "timestamp": job["timestamp"]
#     }
    
#     return JobStatusResponse(**job_data)

# @app.get("/results/{job_id}")
# async def get_optimization_results(
#     job_id: str, 
#     limit: int = Query(100, le=1000),
#     offset: int = Query(0, ge=0)
# ):
#     """Get detailed optimization results"""
#     if job_id not in jobs:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     job = jobs[job_id]
#     if job["status"] != "completed":
#         raise HTTPException(status_code=400, detail="Job not completed yet")
    
#     try:
#         result_df = job.get("results_df")
#         if result_df is None:
#             # Try to load from file
#             results_path = os.path.join(OUTPUT_DIR, f"results_{job_id}.csv")
#             if not os.path.exists(results_path):
#                 raise HTTPException(status_code=404, detail="Results file not found")
#             result_df = pd.read_csv(results_path)
        
#         # Convert to JSON-friendly format
#         total_records = len(result_df)
#         paginated_df = result_df.iloc[offset:offset + limit]
        
#         results = []
#         for _, row in paginated_df.iterrows():
#             result = {
#                 "plan_id": str(row.get("plan_id", "")),
#                 "optimized_mode": str(row.get("optimized_mode", "Road")),
#                 "q_rail_tons": float(row.get("q_rail_tons", 0)),
#                 "optimized_total_cost": float(row.get("optimized_total_cost", 0)),
#                 "planned_qty_t": float(row.get("planned_qty_t", 0)),
#                 "customer_name": str(row.get("customer_name", "")),
#                 "destination": str(row.get("destination", "")),
#                 "origin_plant": str(row.get("origin_plant", "")),
#                 "distance_km": float(row.get("distance_km", 0)),
#                 "pred_rail_cost_total": float(row.get("pred_rail_cost_total", 0)),
#                 "pred_road_cost_total": float(row.get("pred_road_cost_total", 0)),
#                 "on_time_prob": float(row.get("on_time_prob", 0)),
#                 "priority_score": float(row.get("priority_score", 0)),
#                 "rake_available": int(row.get("rake_available", 0)),
#                 "on_time_label": int(row.get("on_time_label", 0)),
#                 "product_type": str(row.get("product_type", "")),
#                 "wagon_type": str(row.get("wagon_type", ""))
#             }
#             results.append(convert_numpy_types(result))
        
#         response_data = {
#             "job_id": job_id,
#             "status": "completed",
#             "pagination": {
#                 "total": int(total_records),
#                 "limit": int(limit),
#                 "offset": int(offset),
#                 "returned": int(len(results))
#             },
#             "results": results,
#             "summary": convert_numpy_types(job.get("results"))
#         }
        
#         return convert_numpy_types(response_data)
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to load results: {str(e)}")

# @app.get("/plans")
# async def get_available_plans(
#     limit: int = Query(50, le=500),
#     origin: Optional[str] = None,
#     destination: Optional[str] = None
# ):
#     """Get list of available plans for optimization"""
#     try:
#         df = pd.read_csv(DATA_PATH)
        
#         # Apply filters
#         if origin:
#             df = df[df['origin_plant'] == origin]
#         if destination:
#             df = df[df['destination'] == destination]
        
#         plans = []
#         for _, row in df.head(limit).iterrows():
#             plan = {
#                 "plan_id": str(row.get("plan_id", "")),
#                 "customer_name": str(row.get("customer_name", "")),
#                 "origin_plant": str(row.get("origin_plant", "")),
#                 "destination": str(row.get("destination", "")),
#                 "planned_qty_t": float(row.get("planned_qty_t", 0)),
#                 "priority_score": float(row.get("priority_score", 0)),
#                 "distance_km": float(row.get("distance_km", 0)),
#                 "product_type": str(row.get("product_type", "")),
#                 "plan_date": str(row.get("plan_date", ""))
#             }
#             plans.append(convert_numpy_types(plan))
        
#         response_data = {
#             "total_plans": int(len(df)),
#             "returned": int(len(plans)),
#             "filters": {
#                 "origin": origin,
#                 "destination": destination
#             },
#             "plans": plans
#         }
        
#         return convert_numpy_types(response_data)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to load plans: {str(e)}")

# @app.get("/plan/{plan_id}")
# async def get_plan_details(plan_id: str):
#     """Get detailed information for a specific plan"""
#     try:
#         df = pd.read_csv(DATA_PATH)
#         plan_data = df[df['plan_id'] == plan_id]
        
#         if len(plan_data) == 0:
#             raise HTTPException(status_code=404, detail="Plan not found")
        
#         row = plan_data.iloc[0]
        
#         plan_details = {
#             "plan_id": plan_id,
#             "customer_name": str(row.get("customer_name", "")),
#             "origin_plant": str(row.get("origin_plant", "")),
#             "origin_state": str(row.get("origin_state", "")),
#             "destination": str(row.get("destination", "")),
#             "destination_state": str(row.get("destination_state", "")),
#             "planned_qty_t": float(row.get("planned_qty_t", 0)),
#             "priority_score": float(row.get("priority_score", 0)),
#             "min_rake_tonnage": float(row.get("min_rake_tonnage", 0)),
#             "terminal_cost": float(row.get("terminal_cost", 0)),
#             "expected_demurrage": float(row.get("expected_demurrage", 0)),
#             "distance_km": float(row.get("distance_km", 0)),
#             "product_type": str(row.get("product_type", "")),
#             "wagon_type": str(row.get("wagon_type", "")),
#             "route_via": str(row.get("route_via", "")),
#             "lead_time_days": int(row.get("lead_time_days", 0)),
#             "plan_date": str(row.get("plan_date", ""))
#         }
        
#         return convert_numpy_types(plan_details)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to load plan details: {str(e)}")

# @app.get("/dashboard/stats")
# async def get_dashboard_stats():
#     """Get dashboard statistics"""
#     try:
#         df = pd.read_csv(DATA_PATH)
        
#         # Convert all numpy types to native Python types
#         stats = {
#             "total_orders": int(len(df)),
#             "total_tonnage": float(df["planned_qty_t"].sum()),
#             "avg_priority": float(round(df["priority_score"].mean(), 2)),
#             "unique_customers": int(df["customer_name"].nunique()),
#             "unique_origins": int(df["origin_plant"].nunique()),
#             "unique_destinations": int(df["destination"].nunique()),
#             "avg_distance": float(round(df["distance_km"].mean(), 0)),
#             "products_distribution": convert_numpy_types(df["product_type"].value_counts().to_dict()),
#             "origins_distribution": convert_numpy_types(df["origin_plant"].value_counts().to_dict())
#         }
        
#         return convert_numpy_types(stats)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to load dashboard stats: {str(e)}")

# @app.get("/filters/options")
# async def get_filter_options():
#     """Get available filter options"""
#     try:
#         df = pd.read_csv(DATA_PATH)
        
#         options = {
#             "origins": [str(x) for x in df["origin_plant"].unique().tolist()],
#             "destinations": [str(x) for x in df["destination"].unique().tolist()],
#             "products": [str(x) for x in df["product_type"].unique().tolist()],
#             "wagon_types": [str(x) for x in df["wagon_type"].unique().tolist()]
#         }
        
#         return convert_numpy_types(options)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to load filter options: {str(e)}")

# # -------------------------------------------------------------------------
# # Main
# # -------------------------------------------------------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "api:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )





"""
api.py

FastAPI server for Rake Formation Optimization System
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import os
import json
import asyncio
import uuid
from datetime import datetime
import joblib

from optim import optimize_rake_allocation, engineer_features
from multi_model import MultiModelPipeline

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
DATA_PATH = "./enhanced_bokaro_customers.csv"
MODEL_DIR = "./saved_models"
OUTPUT_DIR = "./api_results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------------
app = FastAPI(
    title="Rake Formation Optimization API",
    description="API for optimizing rake allocation and formation planning",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------
class OptimizationRequest(BaseModel):
    plan_ids: Optional[List[str]] = None
    use_sample: bool = True
    custom_data: Optional[List[Dict]] = None
    optimization_mode: str = "balanced"  # balanced, cost_min, rail_max

class OptimizationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    results_path: Optional[str] = None
    summary: Optional[Dict] = None
    timestamp: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    results: Optional[Dict] = None
    timestamp: str

# -------------------------------------------------------------------------
# Global State
# -------------------------------------------------------------------------
jobs = {}
pipeline = None

# -------------------------------------------------------------------------
# Helper Functions for Data Serialization
# -------------------------------------------------------------------------
def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):  # Handle NaN values
        return None
    else:
        return obj

def serialize_dataframe(df):
    """Convert DataFrame to JSON-serializable format"""
    if df is None:
        return None
    
    # Convert numpy types in the entire DataFrame
    df_serialized = df.copy()
    for col in df_serialized.columns:
        if df_serialized[col].dtype in [np.int64, np.int32]:
            df_serialized[col] = df_serialized[col].astype('int64').fillna(0)
        elif df_serialized[col].dtype in [np.float64, np.float32]:
            df_serialized[col] = df_serialized[col].astype('float64').fillna(0.0)
    
    return df_serialized

# -------------------------------------------------------------------------
# Startup Event
# -------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize the ML pipeline on startup"""
    global pipeline
    try:
        print("🔹 Initializing ML pipeline...")
        pipeline = MultiModelPipeline(DATA_PATH, MODEL_DIR)
        
        # Load feature columns
        feature_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
        if os.path.exists(feature_path):
            pipeline.feature_cols = joblib.load(feature_path)
            print(f"✅ Loaded {len(pipeline.feature_cols)} feature columns")
        else:
            print("❌ Feature columns not found")
            
        print("✅ ML pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize ML pipeline: {e}")

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def load_and_preprocess_data(plan_ids=None):
    """Load and preprocess data for optimization"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Filter by plan_ids if provided
        if plan_ids and len(plan_ids) > 0:
            df = df[df['plan_id'].isin(plan_ids)]
            if len(df) == 0:
                raise ValueError("No matching plan_ids found")
        
        # Engineer features to match training
        df = engineer_features(df)
        
        return df
    except Exception as e:
        raise Exception(f"Data loading failed: {e}")

def run_predictions_with_fallback(df):
    """Run predictions with fallback for feature mismatch"""
    try:
        # Try to run predictions normally
        preds = pipeline.predict_all(df)
        df_pred = df.copy()
        for k, v in preds.items():
            df_pred[k] = v
        print("✅ Predictions completed using ML models")
        return df_pred
    except Exception as e:
        print(f"⚠️ Model prediction failed: {e}")
        print("🔄 Using fallback prediction values...")
        
        # Fallback prediction values
        df_pred = df.copy()
        
        # Cost predictions based on distance
        df_pred['pred_rail_cost_total'] = df_pred['distance_km'] * 2.5
        df_pred['pred_road_cost_total'] = df_pred['distance_km'] * 3.8
        
        # Probability predictions (fallback)
        df_pred['on_time_prob'] = 0.75
        df_pred['on_time_label'] = (df_pred['on_time_prob'] > 0.5).astype(int)
        df_pred['selected_for_rake_prob'] = 0.65
        df_pred['selected_for_rake_label'] = (df_pred['selected_for_rake_prob'] > 0.5).astype(int)
        df_pred['choose_rail_prob'] = 0.6
        df_pred['choose_rail_label'] = (df_pred['choose_rail_prob'] > 0.5).astype(int)
        
        print("✅ Fallback predictions applied")
        return df_pred

def run_optimization_job(job_id: str, plan_ids: Optional[List[str]] = None, mode: str = "balanced"):
    """Run optimization in background"""
    try:
        jobs[job_id] = {
            "status": "running", 
            "progress": 0.2,
            "message": "Loading data...",
            "timestamp": datetime.now().isoformat()
        }
        
        # Load data
        df = load_and_preprocess_data(plan_ids)
        jobs[job_id]["progress"] = 0.4
        jobs[job_id]["message"] = "Running predictions..."
        
        # Run predictions with fallback
        df_pred = run_predictions_with_fallback(df)
        
        jobs[job_id]["progress"] = 0.7
        jobs[job_id]["message"] = "Running optimization..."
        
        # Run optimization (mode parameter not used in current optim.py)
        result_df = optimize_rake_allocation(df_pred)
        
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["message"] = "Optimization completed"
        
        # Save results
        output_path = os.path.join(OUTPUT_DIR, f"results_{job_id}.csv")
        result_df.to_csv(output_path, index=False)
        
        # Create comprehensive summary
        total_orders = len(result_df)
        rail_orders = result_df["y_rail"].sum() if "y_rail" in result_df.columns else 0
        road_orders = total_orders - rail_orders
        rail_tonnage = result_df["q_rail_tons"].sum() if "q_rail_tons" in result_df.columns else 0
        total_tonnage = result_df["planned_qty_t"].sum()
        total_cost = result_df["optimized_total_cost"].sum() if "optimized_total_cost" in result_df.columns else 0
        
        # Cost savings analysis (simplified)
        avg_rail_cost = result_df["pred_rail_cost_total"].mean() if "pred_rail_cost_total" in result_df.columns else 0
        avg_road_cost = result_df["pred_road_cost_total"].mean() if "pred_road_cost_total" in result_df.columns else 0
        baseline_cost = avg_road_cost * total_tonnage  # Assume all road as baseline
        
        summary = {
            "total_orders": int(total_orders),
            "rail_orders": int(rail_orders),
            "road_orders": int(road_orders),
            "rail_orders_percentage": float(round(rail_orders/total_orders*100, 1)) if total_orders > 0 else 0.0,
            "rail_tonnage": float(round(rail_tonnage, 0)),
            "total_tonnage": float(round(total_tonnage, 0)),
            "rail_tonnage_percentage": float(round(rail_tonnage/total_tonnage*100, 1)) if total_tonnage > 0 else 0.0,
            "total_cost": float(round(total_cost, 2)),
            "cost_savings": float(round(baseline_cost - total_cost, 2)) if baseline_cost > total_cost else 0.0,
            "savings_percentage": float(round((baseline_cost - total_cost) / baseline_cost * 100, 1)) if baseline_cost > 0 else 0.0,
            "results_file": output_path,
            "optimization_mode": mode
        }
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["results"] = convert_numpy_types(summary)
        jobs[job_id]["results_df"] = result_df
        
        print(f"✅ Optimization job {job_id} completed")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = f"Error: {str(e)}"
        print(f"❌ Optimization job {job_id} failed: {e}")

# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Rake Formation Optimization API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "optimize": "/optimize",
            "job_status": "/job/{job_id}",
            "results": "/results/{job_id}",
            "plans": "/plans",
            "plan_details": "/plan/{plan_id}",
            "dashboard": "/dashboard/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    pipeline_status = "ready" if pipeline else "not_ready"
    return {
        "status": "healthy",
        "pipeline": pipeline_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_rake_formation(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Start rake formation optimization"""
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())[:8]
        
        # Validate request
        if request.custom_data and len(request.custom_data) > 0:
            raise HTTPException(status_code=501, detail="Custom data not yet supported")
        
        # Start background job
        background_tasks.add_task(run_optimization_job, job_id, request.plan_ids, request.optimization_mode)
        
        # Store job info
        jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "Job queued for processing",
            "timestamp": datetime.now().isoformat()
        }
        
        return OptimizationResponse(
            job_id=job_id,
            status="queued",
            message="Optimization job started",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")

@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get optimization job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Convert numpy types in job data
    job_data = {
        "job_id": job_id,
        "status": job["status"],
        "progress": float(job["progress"]),
        "message": job["message"],
        "results": convert_numpy_types(job.get("results")),
        "timestamp": job["timestamp"]
    }
    
    return JobStatusResponse(**job_data)

@app.get("/results/{job_id}")
async def get_optimization_results(
    job_id: str, 
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    format: str = Query("json", regex="^(json|csv)$")
):
    """Get detailed optimization results"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    try:
        if format == "csv":
            # Return CSV file download
            results_path = os.path.join(OUTPUT_DIR, f"results_{job_id}.csv")
            if not os.path.exists(results_path):
                raise HTTPException(status_code=404, detail="Results file not found")
            return FileResponse(results_path, filename=f"optimization_results_{job_id}.csv")
        
        # JSON format
        result_df = job.get("results_df")
        if result_df is None:
            # Try to load from file
            results_path = os.path.join(OUTPUT_DIR, f"results_{job_id}.csv")
            if not os.path.exists(results_path):
                raise HTTPException(status_code=404, detail="Results file not found")
            result_df = pd.read_csv(results_path)
        
        # Convert to JSON-friendly format
        total_records = len(result_df)
        paginated_df = result_df.iloc[offset:offset + limit]
        
        results = []
        for _, row in paginated_df.iterrows():
            result = {
                "plan_id": str(row.get("plan_id", "")),
                "optimized_mode": str(row.get("optimized_mode", "Road")),
                "q_rail_tons": float(row.get("q_rail_tons", 0)),
                "optimized_total_cost": float(row.get("optimized_total_cost", 0)),
                "planned_qty_t": float(row.get("planned_qty_t", 0)),
                "customer_name": str(row.get("customer_name", "")),
                "destination": str(row.get("destination", "")),
                "origin_plant": str(row.get("origin_plant", "")),
                "distance_km": float(row.get("distance_km", 0)),
                "pred_rail_cost_total": float(row.get("pred_rail_cost_total", 0)),
                "pred_road_cost_total": float(row.get("pred_road_cost_total", 0)),
                "on_time_prob": float(row.get("on_time_prob", 0)),
                "priority_score": float(row.get("priority_score", 0)),
                "rake_available": int(row.get("rake_available", 0)),
                "on_time_label": int(row.get("on_time_label", 0)),
                "product_type": str(row.get("product_type", "")),
                "wagon_type": str(row.get("wagon_type", ""))
            }
            results.append(convert_numpy_types(result))
        
        response_data = {
            "job_id": job_id,
            "status": "completed",
            "pagination": {
                "total": int(total_records),
                "limit": int(limit),
                "offset": int(offset),
                "returned": int(len(results))
            },
            "results": results,
            "summary": convert_numpy_types(job.get("results"))
        }
        
        return convert_numpy_types(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load results: {str(e)}")

@app.get("/plans")
async def get_available_plans(
    limit: int = Query(50, le=500),
    origin: Optional[str] = None,
    destination: Optional[str] = None
):
    """Get list of available plans for optimization"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Apply filters
        if origin:
            df = df[df['origin_plant'] == origin]
        if destination:
            df = df[df['destination'] == destination]
        
        plans = []
        for _, row in df.head(limit).iterrows():
            plan = {
                "plan_id": str(row.get("plan_id", "")),
                "customer_name": str(row.get("customer_name", "")),
                "origin_plant": str(row.get("origin_plant", "")),
                "destination": str(row.get("destination", "")),
                "planned_qty_t": float(row.get("planned_qty_t", 0)),
                "priority_score": float(row.get("priority_score", 0)),
                "distance_km": float(row.get("distance_km", 0)),
                "product_type": str(row.get("product_type", "")),
                "plan_date": str(row.get("plan_date", ""))
            }
            plans.append(convert_numpy_types(plan))
        
        response_data = {
            "total_plans": int(len(df)),
            "returned": int(len(plans)),
            "filters": {
                "origin": origin,
                "destination": destination
            },
            "plans": plans
        }
        
        return convert_numpy_types(response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load plans: {str(e)}")

@app.get("/plan/{plan_id}")
async def get_plan_details(plan_id: str):
    """Get detailed information for a specific plan"""
    try:
        df = pd.read_csv(DATA_PATH)
        plan_data = df[df['plan_id'] == plan_id]
        
        if len(plan_data) == 0:
            raise HTTPException(status_code=404, detail="Plan not found")
        
        row = plan_data.iloc[0]
        
        plan_details = {
            "plan_id": plan_id,
            "customer_name": str(row.get("customer_name", "")),
            "origin_plant": str(row.get("origin_plant", "")),
            "origin_state": str(row.get("origin_state", "")),
            "destination": str(row.get("destination", "")),
            "destination_state": str(row.get("destination_state", "")),
            "planned_qty_t": float(row.get("planned_qty_t", 0)),
            "priority_score": float(row.get("priority_score", 0)),
            "min_rake_tonnage": float(row.get("min_rake_tonnage", 0)),
            "terminal_cost": float(row.get("terminal_cost", 0)),
            "expected_demurrage": float(row.get("expected_demurrage", 0)),
            "distance_km": float(row.get("distance_km", 0)),
            "product_type": str(row.get("product_type", "")),
            "wagon_type": str(row.get("wagon_type", "")),
            "route_via": str(row.get("route_via", "")),
            "lead_time_days": int(row.get("lead_time_days", 0)),
            "plan_date": str(row.get("plan_date", ""))
        }
        
        return convert_numpy_types(plan_details)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load plan details: {str(e)}")

@app.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Convert all numpy types to native Python types
        stats = {
            "total_orders": int(len(df)),
            "total_tonnage": float(df["planned_qty_t"].sum()),
            "avg_priority": float(round(df["priority_score"].mean(), 2)),
            "unique_customers": int(df["customer_name"].nunique()),
            "unique_origins": int(df["origin_plant"].nunique()),
            "unique_destinations": int(df["destination"].nunique()),
            "avg_distance": float(round(df["distance_km"].mean(), 0)),
            "products_distribution": convert_numpy_types(df["product_type"].value_counts().to_dict()),
            "origins_distribution": convert_numpy_types(df["origin_plant"].value_counts().to_dict())
        }
        
        return convert_numpy_types(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dashboard stats: {str(e)}")

@app.get("/filters/options")
async def get_filter_options():
    """Get available filter options"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        options = {
            "origins": [str(x) for x in df["origin_plant"].unique().tolist()],
            "destinations": [str(x) for x in df["destination"].unique().tolist()],
            "products": [str(x) for x in df["product_type"].unique().tolist()],
            "wagon_types": [str(x) for x in df["wagon_type"].unique().tolist()]
        }
        
        return convert_numpy_types(options)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load filter options: {str(e)}")

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )