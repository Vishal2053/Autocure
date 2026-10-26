import pickle
import pandas as pd

def serve(model_or_result, host: str = "127.0.0.1", port: int = 8000):
    """
    1-Function Model Server:
    Instantly deploy any trained autocure model as a production FastAPI REST API.
    
    Includes:
    - POST /predict: Accepts a JSON list of records or single record and returns predictions
    - GET /health: Health check endpoint
    - Interactive Swagger documentation at http://<host>:<port>/docs
    """
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
        from typing import List, Dict, Any, Union
    except ImportError:
        raise ImportError(
            "FastAPI and Uvicorn are required for ac.serve(). "
            "Install them with: pip install fastapi uvicorn"
        )

    # Extract model
    if isinstance(model_or_result, str):
        with open(model_or_result, "rb") as f:
            model = pickle.load(f)
    elif hasattr(model_or_result, "best_model"):
        model = model_or_result.best_model
    elif isinstance(model_or_result, dict) and "best_model" in model_or_result:
        model = model_or_result["best_model"]
    else:
        model = model_or_result

    app = FastAPI(
        title="Autocure Model API",
        description="Auto-generated high-performance REST API for your trained machine learning model.",
        version="1.0.0"
    )

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "service": "autocure-api"}

    @app.post("/predict")
    def predict(data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        try:
            if isinstance(data, dict):
                df_input = pd.DataFrame([data])
            else:
                df_input = pd.DataFrame(data)

            preds = model.predict(df_input).tolist()
            response = {"predictions": preds}

            # If predict_proba is available
            estimator = getattr(model, "named_steps", {}).get("clf", model)
            if hasattr(estimator, "predict_proba") and hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(df_input).tolist()
                    response["probabilities"] = probs
                except Exception:
                    pass

            return response
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    print(f"\n[Autocure Serve] Launching API server at http://{host}:{port}")
    print(f"[Autocure Serve] Interactive API Docs available at http://{host}:{port}/docs\n")
    uvicorn.run(app, host=host, port=port)
