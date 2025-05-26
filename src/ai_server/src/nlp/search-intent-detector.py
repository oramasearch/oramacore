import spacy
import threading
import time
from typing import Dict, Optional
from collections import OrderedDict
import logging


class LazySpacyModelManager:
    """
    Manages spaCy models with lazy loading and automatic memory cleanup.

    Features:
    - Lazy loading: Models loaded only when first requested
    - LRU eviction: Automatically removes least recently used models
    - Time-based eviction: Removes models after inactivity period
    - Memory efficient: Only keeps active models in memory
    - Thread-safe: Can be used in multi-threaded applications
    """

    def __init__(
        self,
        max_models: int = 10,  # Maximum models to keep in memory at the same time
        eviction_time: int = 600,  # 10 minutes
        cleanup_interval: int = 60,  # 1 minute
        default_model: str = "en_core_web_sm",
    ):
        """
        Initialize the model manager.

        Args:
            max_models: Maximum number of models to keep in memory
            eviction_time: Seconds of inactivity before model is eligible for eviction
            cleanup_interval: Seconds between cleanup cycles
            default_model: Fallback model if requested model fails to load
        """
        self.max_models = max_models
        self.eviction_time = eviction_time
        self.cleanup_interval = cleanup_interval
        self.default_model = default_model

        # Thread-safe storage
        self._models: OrderedDict[str, spacy.Language] = OrderedDict()
        self._last_used: Dict[str, float] = {}
        self._lock = threading.RLock()

        # Model mapping for different languages
        self.model_map = {
            "en": "en_core_web_sm",
            "es": "es_core_news_sm",
            "fr": "fr_core_news_sm",
            "de": "de_core_news_sm",
            "it": "it_core_news_sm",
            "pt": "pt_core_news_sm",
            "nl": "nl_core_news_sm",
            "ru": "ru_core_news_sm",
            "zh": "zh_core_web_sm",
            "ja": "ja_core_news_sm",
            "ko": "ko_core_news_sm",
            "da": "da_core_news_sm",
            "sv": "sv_core_news_sm",
            "nb": "nb_core_news_sm",
            "fi": "fi_core_news_sm",
            "pl": "pl_core_news_sm",
            "ro": "ro_core_news_sm",
            "el": "el_core_news_sm",
            "hr": "hr_core_news_sm",
            "sl": "sl_core_news_sm",
            "uk": "uk_core_news_sm",
            "lt": "lt_core_news_sm",
            "mk": "mk_core_news_sm",
            "ca": "ca_core_news_sm",
        }

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()

        self.logger = logging.getLogger(__name__)

    def get_model(self, lang_code: str) -> spacy.Language:
        """
        Get a spaCy model for the given language code.
        Loads the model if not in memory, manages LRU cache.

        Args:
            lang_code: Language code (e.g., 'en', 'es', 'fr')

        Returns:
            Loaded spaCy model
        """
        model_name = self.model_map.get(lang_code, self.default_model)

        with self._lock:
            # Update access time
            self._last_used[model_name] = time.time()

            # Return if already loaded
            if model_name in self._models:
                # Move to end (most recently used)
                self._models.move_to_end(model_name)
                self.logger.debug(f"Using cached model: {model_name}")
                return self._models[model_name]

            # Load new model
            model = self._load_model(model_name)

            # Add to cache
            self._models[model_name] = model
            self._models.move_to_end(model_name)

            # Evict if over limit
            self._evict_excess_models()

            self.logger.info(f"Loaded new model: {model_name}")
            return model

    def _load_model(self, model_name: str) -> spacy.Language:
        """Load a spaCy model with fallback to default."""
        try:
            return spacy.load(model_name)
        except OSError:
            self.logger.warning(f"Model {model_name} not found, using default: {self.default_model}")
            try:
                return spacy.load(self.default_model)
            except OSError:
                raise RuntimeError(f"Neither {model_name} nor default model {self.default_model} could be loaded")

    def _evict_excess_models(self):
        """Remove excess models to stay within max_models limit."""
        while len(self._models) > self.max_models:
            # Remove least recently used (first in OrderedDict)
            oldest_model = next(iter(self._models))
            del self._models[oldest_model]
            if oldest_model in self._last_used:
                del self._last_used[oldest_model]
            self.logger.info(f"Evicted model due to size limit: {oldest_model}")

    def _cleanup_worker(self):
        """Background thread that periodically cleans up unused models."""
        while True:
            time.sleep(self.cleanup_interval)
            self._cleanup_stale_models()

    def _cleanup_stale_models(self):
        """Remove models that haven't been used recently."""
        current_time = time.time()
        stale_models = []

        with self._lock:
            for model_name, last_used in self._last_used.items():
                if current_time - last_used > self.eviction_time:
                    stale_models.append(model_name)

            for model_name in stale_models:
                if model_name in self._models:
                    del self._models[model_name]
                del self._last_used[model_name]
                self.logger.info(f"Evicted stale model: {model_name}")

    def preload_models(self, lang_codes: list[str]):
        """
        Preload models for specific languages.
        Useful for warming up cache with expected languages.
        """
        for lang_code in lang_codes:
            self.get_model(lang_code)
            self.logger.info(f"Preloaded model for language: {lang_code}")

    def get_stats(self) -> Dict:
        """Get current manager statistics."""
        with self._lock:
            return {
                "loaded_models": list(self._models.keys()),
                "model_count": len(self._models),
                "max_models": self.max_models,
                "last_used_times": {model: time.time() - last_used for model, last_used in self._last_used.items()},
            }

    def clear_cache(self):
        """Manually clear all cached models."""
        with self._lock:
            self._models.clear()
            self._last_used.clear()
            self.logger.info("Cleared all cached models")


class LazySearchIntentDetector:
    """
    Search intent detector with lazy model loading and automatic memory management.
    """

    def __init__(self, max_models: int = 6, eviction_time: int = 600, default_lang: str = "en"):  # 10 minutes
        self.model_manager = LazySpacyModelManager(max_models=max_models, eviction_time=eviction_time)
        self.default_lang = default_lang

        # Performance tracking
        self.request_count = 0
        self.cache_hits = 0

    def extract_searchable_content(self, text: str, lang: str = None) -> Optional[str]:
        """
        Extract searchable content with automatic model management.

        Args:
            text: Input text to analyze
            lang: Language code (auto-detects if None)

        Returns:
            Searchable content or None
        """
        self.request_count += 1

        # Use default language if not specified
        if lang is None:
            lang = self.default_lang

        # Get model (lazy loaded)
        nlp = self.model_manager.get_model(lang)

        # Process text using your existing logic
        doc = nlp(text.strip())

        # Find substantial noun chunks
        searchable_chunks = []
        for chunk in doc.noun_chunks:
            if self._is_substantial_content(chunk):
                searchable_chunks.append(chunk.text.strip())

        # Find objects from structure
        for token in doc:
            if token.pos_ == "VERB":
                verb_objects = self._get_verb_objects(token)
                for obj in verb_objects:
                    if self._is_substantial_content(obj):
                        searchable_chunks.append(obj.text.strip())

        # Return best candidate
        if searchable_chunks:
            return max(set(searchable_chunks), key=len)

        return None

    def should_search(self, text: str, lang: str = None) -> bool:
        """Check if text contains searchable content."""
        return self.extract_searchable_content(text, lang) is not None

    def _is_substantial_content(self, span) -> bool:
        """Check if span contains substantial searchable content."""
        content_words = [
            token
            for token in span
            if token.pos_ in {"NOUN", "PROPN", "ADJ", "NUM"} and not token.is_stop and not token.is_punct
        ]
        return len(content_words) >= 1

    def _get_verb_objects(self, verb_token):
        """Extract objects related to a verb."""
        objects = []
        for child in verb_token.children:
            if child.dep_ in {"dobj", "iobj", "obj", "obl", "nmod"}:
                start = child.left_edge.i
                end = child.right_edge.i + 1
                objects.append(child.doc[start:end])
        return objects

    def get_stats(self) -> Dict:
        """Get performance and cache statistics."""
        manager_stats = self.model_manager.get_stats()
        return {
            **manager_stats,
            "total_requests": self.request_count,
            "cache_hit_rate": (self.cache_hits / self.request_count * 100) if self.request_count > 0 else 0,
        }
