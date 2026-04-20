from .helper import api_model_predict, api_model_feedback, api_model_get_prediction, get_prediction_json_schema

import os

API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://localhost:{API_PORT}"


def test_api_model_predict_feedback():
    prediction_id = api_model_predict()
    
    api_model_feedback(prediction_id, "negative")

    api_model_get_prediction(prediction_id, with_feedback=True)



    


    
    