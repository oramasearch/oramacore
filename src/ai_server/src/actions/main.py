import json
import logging
import requests
from typing import Dict

from src.utils import OramaAIConfig

logger = logging.getLogger(__name__)


class Actions:
    def __init__(self, config: OramaAIConfig):
        self.config = config

    def call_oramacore_search(self, collection_id: str, query: Dict, api_key: str):
        body = json.dumps(query)
        url = f"http://{self.config.rust_server_host}:{self.config.rust_server_port}/v1/{collection_id}/actions/execute?api-key={api_key}"
        headers = {"Content-Type": "application/json; charset=utf-8"}

        try:
            resp = requests.post(
                url=url, json={"context": body, "name": "search"}, headers=headers
            )
            as_json = json.loads(resp.text)
            return as_json["hits"]

        except Exception as e:
            print(e)
            logger.error(f"Error calling oramacore search: {e}")
            return None
