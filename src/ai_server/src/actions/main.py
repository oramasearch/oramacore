import logging
import requests
from typing import Dict

from src.utils import OramaAIConfig

logger = logging.getLogger(__name__)


class Actions:
    def __init__(self, config: OramaAIConfig):
        self.config = config

    def call_oramacore_search(self, collection_id: str, query: Dict) -> any:
        url = f"http://localhost:8080/v0/{collection_id}/actions/execute"  # @todo: take base url and port from config
        headers = {"Content-Type": "application/json; charset=utf-8"}
        resp = requests.post(url=url, json=query, headers=headers)

        resp.json()
