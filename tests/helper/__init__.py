from .TestEndpoint import TestEndpoint

import os

import uuid

API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://localhost:{API_PORT}"

def get_model_labels():
    return ["positive", "negative", "neutral"]

def get_feedback_json_schema():
    return {
        "type": "object",
        "properties": {
            "true_label": {
                "type": "string",
                "enum": get_model_labels()
            },
            "timestamp": {
                "type": "string",
                "format": "date-time"
            }
        },
        "required": ["true_label", "timestamp"]
    }

def get_prediction_json_schema(with_feedback: bool | None = None):
    json_schema = {
                "type": "object",
                "properties": {
                    "prediction_id": {
                        "type": "string",
                        "format": "uuid"
                    },
                    "input_text": {
                        "type": "string"
                    },
                    "predicted_label": {
                        "type": "string",
                        "enum": get_model_labels()
                    },
                    "confidence": {
                        "type": "number"
                    }
                },
                "required": ["prediction_id", "input_text", "predicted_label", "confidence"]
            }

    if with_feedback is not None:
        json_schema["properties"]["feedback"] = get_feedback_json_schema() if with_feedback else {"type": "null"}
        json_schema["required"].append("feedback")
    
    return json_schema

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
                

def api_status_up(model_loaded: bool | None = None):
    print("# Testing API status...")

    method = "GET"
    url = BASE_URL+"/status"
    json_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["ok"]
            }
        },
        "required": ["status"]
    }

    if model_loaded is None:
        json_schema["properties"]["model"] = {
            "oneOf": [
                get_model_json_schema(True),
                get_model_json_schema(False)
            ]
        }
        json_schema["required"].append("model")
    elif isinstance(model_loaded, bool):
        json_schema["properties"]["model"] = get_model_json_schema(model_loaded)
        json_schema["required"].append("model")
    else:
        raise ValueError("model_loaded must be a boolean or None")
        
    TestEndpoint(url=url, method=method).test(status_code=200, json_schema=json_schema)

def api_load_model(already_loaded: bool | None = None):
    print("# Testing model load...")

    method = "GET"
    url = BASE_URL+"/load-model"

    json_schema_model_already_loaded = {
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
    json_schema_model_not_already_loaded = {
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

    json_schema = {}

    if already_loaded is None:
        json_schema = {
            "oneOf": [
                json_schema_model_already_loaded,
                json_schema_model_not_already_loaded
            ]
        }
    elif already_loaded is True:
        json_schema = json_schema_model_already_loaded
    elif already_loaded is False:
        json_schema = json_schema_model_not_already_loaded
        
    TestEndpoint(url=url, method=method).test(status_code=200, json_schema=json_schema)
    
def api_unload_model(already_unloaded: bool | None = None):
    print("# Testing model unload...")

    method = "GET"
    url = BASE_URL+"/unload-model"

    json_schema = {}

    if already_unloaded is True:
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
    elif already_unloaded is False:
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
    TestEndpoint(url=url, method=method).test(status_code=200, json_schema=json_schema)

def verify_model_loaded() -> bool:
    print("# Verify model is loaded...")

    try:
        api_status_up(model_loaded=True)
    except Exception as e:
        print(f"--> Error: model not loaded!")
        print(f"## Loading model...")

        try:
            api_load_model(already_loaded=False)
        except Exception as e:
            print(f"--> Error: impossible to load model!")
            return False

    print("--> Model is loaded")
    return True

def api_model_predict() -> uuid.UUID:
    print("# Testing model predict...")

    if not verify_model_loaded():
        raise Exception("--> Error: model not loaded!")

    method = "POST"
    url = BASE_URL+"/model/predict"
    data_to_send = {
        "text": "I love this product? Nope"
    }
    
    json_schema = get_prediction_json_schema()

    t = TestEndpoint(url=url, method=method)
    t.test(data=data_to_send, status_code=200, json_schema=json_schema)

    json_response = t.get_response().json()

    prediction_id = json_response["prediction_id"]
    return prediction_id

def api_model_feedback(prediction_id: uuid.UUID, label: str):
    print("# Testing model feedback...")

    if prediction_id is None:
        print("--> Error: prediction_id is None!")
        return

    method = "POST"
    url = BASE_URL+f"/model/predictions/{prediction_id}/feedback"
    data_to_send = {
        "label": label
    }
    json_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["feedback received"]
            },
            "message": {
                "type": "string"
            },
            "prediction_id": {
                "type": "string",
                "format": "uuid",
                "enum": [str(prediction_id)]
            },
            "label": {
                "type": "string",
                "enum": [data_to_send["label"]]
            }
        },
        "required": ["status", "message", "prediction_id", "label"]
    }
    TestEndpoint(url=url, method=method).test(data=data_to_send, status_code=200, json_schema=json_schema)
    
def api_model_get_prediction(prediction_id: str, with_feedback: bool=False):
    print("# Testing model get prediction...")

    if prediction_id is None:
        print("--> Error: prediction_id is None!")
        return

    method = "GET"
    url = BASE_URL+f"/logs/predictions/{prediction_id}"
    json_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["ok"]
            },
            "prediction": get_prediction_json_schema(with_feedback)
        },
        "required": ["status", "prediction"]
    }
    TestEndpoint(url=url, method=method).test(status_code=200, json_schema=json_schema)
    
    