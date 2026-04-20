import requests
from jsonschema import validate, ValidationError, FormatChecker

class TestEndpoint:
    def __init__(self, url: str, method: str, status_code: int, json_schema: dict | None = None):
        self._url = url
        self._method = method
        self._status_code = status_code
        self._json_schema = json_schema
        self._response = None

    def call_endpoint(self):
        self._response = requests.request(self._method, self._url)
        return self

    def validate_status_code(self):
        assert self._response.status_code == self._status_code, f"Expected status code {self._status_code}, got {self._response.status_code}"
        return self

    def validate_response_json(self):
        if self._json_schema is None or self._json_schema == {}:
            return self

        json_data = self._response.json()

        assert isinstance(json_data, dict), f"Expected response JSON to be a dictionary, got {type(json_data)}"

        try:
            validate(instance=json_data, schema=self._json_schema, format_checker=FormatChecker())
        except ValidationError as e:
            raise AssertionError(f"JSON validation failed: {e}")

    def test(self):
        print(f"\n# Testing API status on {self._method} {self._url}...\n")

        self.call_endpoint()
        self.validate_status_code()
        self.validate_response_json()