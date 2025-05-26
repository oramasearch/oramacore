import os
from typing import List


def get_stop_words(lang: str) -> List[str]:
    path = os.path.join(os.path.dirname(__file__), f"../../../nlp/stop_words/{lang}.txt")
    try:
        with open(path) as file:
            return file.read().splitlines()
    except FileNotFoundError:
        return []
    except Exception:
        return []
