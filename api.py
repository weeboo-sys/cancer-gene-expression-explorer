# api.py
from fastapi import FastAPI
import pandas as pd
from pipeline import run_pipeline

app = FastAPI()

@app.post("/analyze")
def analyze(data: dict):
    df = pd.DataFrame(data["records"])
    k = data.get("k", 3)

    result = run_pipeline(df, k)
    return result