from .helper import api_model_predict, api_model_feedback, api_model_get_prediction, get_prediction_json_schema

import os
import time

API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://localhost:{API_PORT}"


def test_api_model_predict_feedback():
    prediction_id = api_model_predict()

    # Attendo 1 secondo per permettere il salvataggio del log della prediction in background
    # necessario per poter associare un feedback ad una prediction
    # (vedi api/routes/model.py endpoint /predict)
    time.sleep(1)
    api_model_feedback(prediction_id, "negative")

    api_model_get_prediction(prediction_id, with_feedback=True)



    


    
    