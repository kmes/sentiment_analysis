from .TestEndpoint import TestEndpoint

import os

API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://localhost:{API_PORT}"

def get_model_json_schema(model_loaded: bool=True):
    return {
                "type": "object",
                "properties": {
                    "loaded": {
                        "type": "boolean",
                        "enum": [model_loaded]
                    },
                    "load_id": {
                        "oneOf": [
                            {"type": "string", "format": "uuid"} if model_loaded else {"type": "null"}
                        ]
                    },
                    "name": {
                        "type": "string"
                    },
                    "version": {
                        "type": "string"
                    }
                },
                "required": ["loaded", "load_id", "name", "version"]
            }
                

def api_status_up(model_loaded: bool=False):
    method = "GET"
    url = BASE_URL+"/status"
    json_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["ok"]
            },
            "model": get_model_json_schema(model_loaded)
        },
        "required": ["status", "model"]
    }
    TestEndpoint(url=url, method=method, status_code=200, json_schema=json_schema).test()

def api_load_model(already_loaded: bool=False):
    method = "GET"
    url = BASE_URL+"/load-model"

    json_schema = {}
    
    if already_loaded:
        json_schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["already loaded"]
                },
                "message": {
                    "type": "string"
                }
            },
            "required": ["status", "message"]
        }
    else:
        json_schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["model loaded"]
                },
                "message": {
                    "type": "string"
                },
                "model": get_model_json_schema(True),
                "time": {
                    "type": "integer"
                }
            },
            "required": ["status", "message", "model", "time"]
        }
    TestEndpoint(url=url, method=method, status_code=200, json_schema=json_schema).test()
    
def api_unload_model(already_unloaded: bool=False):
    method = "GET"
    url = BASE_URL+"/unload-model"

    json_schema = {}

    if already_unloaded:
        json_schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["already unloaded"]
                },
                "message": {
                    "type": "string"
                }
            },
            "required": ["status", "message"]
        }
    else:
        json_schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["model not loaded"]
                },
                "message": {
                    "type": "string"
                },
                "model": get_model_json_schema(False),
                "time": {
                    "type": "integer"
                }
            },
            "required": ["status", "message", "model", "time"]
        }
    TestEndpoint(url=url, method=method, status_code=200, json_schema=json_schema).test()