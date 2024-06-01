import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
#print(sys.path)
from typing import Any

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import random
import numpy as np
import pandas as pd
import joblib

import prometheus_client as prom
from sklearn.metrics import r2_score

# load trained model
save_file_name = "xgboost-model.pkl"
trained_model = joblib.load(filename=save_file_name)


curr_path = str(Path(__file__).parent)

app = FastAPI(
    #title=settings.PROJECT_NAME, 
    #openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

root_router = APIRouter()


################################# Prometheus related code START ######################################################

# Metric object of type gauge
r2_metric = prom.Gauge('r2_score', 'R2 score for random 100 test samples')


# LOAD TEST DATA
test_data = pd.read_csv(curr_path + "/test_dataset.csv")


# Function for updating metrics
def update_metrics():
    test = test_data.sample(100)
    test_feat = test.drop('DEATH_EVENT', axis=1)
    test_cnt = test['DEATH_EVENT'].values
    test_pred = trained_model.predict(test_feat)
    r2 = r2_score(test_cnt, test_pred).round(3)
    r2_metric.set(r2)


@root_router.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())

################################# Prometheus related code END ######################################################


@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )
    return HTMLResponse(content=body)

app.include_router(root_router)


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
