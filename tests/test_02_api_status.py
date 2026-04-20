from .helper import api_status_up, api_load_model, api_unload_model

import os

API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://localhost:{API_PORT}"

def test_api_status():
    api_status_up(model_loaded=False)
    
def test_api_model_load():
    api_load_model(already_loaded=False)

def test_api_model_load_already_loaded():
    api_load_model(already_loaded=True)
    
def test_api_status_model_loaded():
    api_status_up(model_loaded=True)
    
def test_api_model_unload():
    api_unload_model(already_unloaded=False)

def test_api_model_unload_already_unloaded():
    api_unload_model(already_unloaded=True)
    
def test_api_status_model_unloaded():
    api_status_up(model_loaded=False)
    