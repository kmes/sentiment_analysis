import requests
from jsonschema import validate, ValidationError, FormatChecker

class TestEndpoint:
    def __init__(self, url: str, method: str):
        self._url = url
        self._method = method
        self._response = None

    def get_response(self):
        return self._response

    def call_endpoint(self, data: dict | None = None, params: dict | None = None):
        self._response = requests.request(
            method=self._method, 
            url=self._url, 
            params=params,
            json=data
        )
        return self

    def validate_status_code(self, status_code: int = 200):
        assert self._response.status_code == status_code, f"Expected status code {status_code}, got {self._response.status_code}"
        return self

    def validate_response_json(self, json_schema: dict | None = None):
        if json_schema is None or json_schema == {}:
            return self

        json_data = self._response.json()

        assert isinstance(json_data, dict), f"Expected response JSON to be a dictionary, got {type(json_data)}"

        try:
            validate(instance=json_data, schema=json_schema, format_checker=FormatChecker())
        except ValidationError as e:
            raise AssertionError(f"JSON validation failed: {e}")

    def test(self, data: dict | None = None, params: dict | None = None, status_code: int = 200, json_schema: dict | None = None):
        print(f"# Calling API on {self._method} {self._url}...")

        self.call_endpoint(data=data, params=params)
        self.validate_status_code(status_code=status_code)
        self.validate_response_json(json_schema=json_schema)

        return self