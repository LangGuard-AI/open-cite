
import logging
import requests
from typing import Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

class OpenCiteHttpClient:
    """
    Centralized HTTP client for OpenCite.
    Wraps requests.Session to provide consistent headers, logging, and timeouts.
    """
    
    _instance = None

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "OpenCite/1.0",
        })
        self._timeout = 30  # Default timeout in seconds

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Execute an HTTP request.
        """
        # Ensure timeout is set if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = self._timeout

        # Log request details (debug level)
        logger.debug(f"HTTP {method} {url}")
        
        try:
            response = self._session.request(method, url, **kwargs)
            logger.debug(f"HTTP Response: {response.status_code} for {url}")
            return response
        except requests.RequestException as e:
            logger.warning(f"HTTP Request failed: {method} {url} - {e}")
            raise

    def get(self, url: str, **kwargs) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> requests.Response:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs) -> requests.Response:
        return self.request("DELETE", url, **kwargs)

# Global accessor
def get_http_client() -> OpenCiteHttpClient:
    return OpenCiteHttpClient.get_instance()
