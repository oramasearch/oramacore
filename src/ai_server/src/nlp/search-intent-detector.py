import spacy
import threading
import time
import logging
from typing import Dict, Optional, Tuple
from collections import OrderedDict
from src.nlp.stop_words import get_stop_words

try:
    from langdetect import detect_langs, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException

    LANGDETECT_AVAILABLE = True
    DetectorFactory.seed = 0
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from polyglot.detect import Detector as PolyglotDetector

    POLYGLOT_AVAILABLE = True
except ImportError:
    POLYGLOT_AVAILABLE = False

import re


class LanguageDetector:
    def __init__(self, method: str = "auto", confidence_threshold: float = 0.8):
        """
        Initialize language detector.

        Args:
            method: Detection method ("langdetect", "polyglot", "heuristic", "auto")
            confidence_threshold: Minimum confidence to trust detection
        """
        self.method = method
        self.confidence_threshold = confidence_threshold

        # Language code mapping (detection -> spaCy model codes)
        self.lang_map = {
            # Common mappings
            "en": "en",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "it": "it",
            "pt": "pt",
            "nl": "nl",
            "ru": "ru",
            "zh": "zh",
            "ja": "ja",
            "ko": "ko",
            "da": "da",
            "sv": "sv",
            "no": "nb",
            "fi": "fi",
            "pl": "pl",
            "ro": "ro",
            "el": "el",
            "hr": "hr",
            "sl": "sl",
            "uk": "uk",
            "lt": "lt",
            "mk": "mk",
            "ca": "ca",
            # Alternative codes
            "zh-cn": "zh",
            "zh-tw": "zh",
            "nb": "nb",
            "nn": "nb",
            "sr": "hr",  # Serbian -> Croatian (similar)
            "bg": "ru",  # Bulgarian -> Russian (Cyrillic fallback)
            "cs": "pl",  # Czech -> Polish (similar)
            "sk": "pl",  # Slovak -> Polish (similar)
        }

        # Character-based language patterns for heuristic detection
        self.char_patterns = {
            "zh": re.compile(r"[\u4e00-\u9fff]+"),  # Chinese characters
            "ja": re.compile(r"[\u3040-\u309f\u30a0-\u30ff]+"),  # Hiragana/Katakana
            "ko": re.compile(r"[\uac00-\ud7af]+"),  # Hangul
            "ru": re.compile(r"[\u0400-\u04ff]+"),  # Cyrillic
            "el": re.compile(r"[\u0370-\u03ff]+"),  # Greek
            "ar": re.compile(r"[\u0600-\u06ff]+"),  # Arabic (not supported in spaCy)
        }

        # Common words for quick language detection
        self.word_patterns = {
            "en": get_stop_words("en"),
            "es": get_stop_words("es"),
            "fr": get_stop_words("fr"),
            "de": get_stop_words("de"),
            "it": get_stop_words("it"),
            "pt": get_stop_words("pt"),
            "ru": get_stop_words("ru"),
            "nl": get_stop_words("nl"),
            "zh": get_stop_words("zh"),
            "ja": get_stop_words("ja"),
            "ko": get_stop_words("ko"),
            "da": get_stop_words("da"),
            "sv": get_stop_words("sv"),
            "nb": get_stop_words("no"),
            "fi": get_stop_words("fi"),
            "pl": get_stop_words("pl"),
            "ro": get_stop_words("ro"),
            "el": get_stop_words("el"),
            "hr": get_stop_words("hr"),
            "sl": get_stop_words("sl"),
            "uk": get_stop_words("uk"),
            "lt": get_stop_words("lt"),
            "mk": get_stop_words("mk"),
            "ca": get_stop_words("ca"),
        }

        # Initialize detection method
        self._init_detector()

    def _init_detector(self):
        """Initialize the selected detection method."""
        if self.method == "auto":
            if LANGDETECT_AVAILABLE:
                self.method = "langdetect"
            elif POLYGLOT_AVAILABLE:
                self.method = "polyglot"
            else:
                self.method = "heuristic"
                logging.warning("No advanced language detection library available. Using heuristic detection.")

        elif self.method == "langdetect" and not LANGDETECT_AVAILABLE:
            raise ImportError("langdetect not available. Install with: pip install langdetect")
        elif self.method == "polyglot" and not POLYGLOT_AVAILABLE:
            raise ImportError("polyglot not available. Install with: pip install polyglot")

    def detect_language(self, text: str, default: str = "en") -> Tuple[str, float]:
        if not text or len(text.strip()) < 3:
            return default, 1.0

        text = text.strip()

        try:
            if self.method == "langdetect":
                return self._detect_langdetect(text, default)
            elif self.method == "polyglot":
                return self._detect_polyglot(text, default)
            elif self.method == "heuristic":
                return self._detect_heuristic(text, default)
        except Exception as e:
            logging.warning(f"Language detection failed: {e}")

        return default, 1.0

    def _detect_langdetect(self, text: str, default: str) -> Tuple[str, float]:
        try:
            # Get detailed results with probabilities
            results = detect_langs(text)
            if results:
                detected_lang = results[0].lang
                confidence = results[0].prob

                # Map to our supported languages
                mapped_lang = self.lang_map.get(detected_lang, default)

                # Return result if confidence is high enough
                if confidence >= self.confidence_threshold:
                    return mapped_lang, confidence

        except LangDetectException:
            pass

        return default, 1.0

    def _detect_polyglot(self, text: str, default: str) -> Tuple[str, float]:
        """Detect language using Polyglot library."""
        try:
            detector = PolyglotDetector(text)
            detected_lang = detector.language.code
            confidence = detector.language.confidence / 100.0  # Convert to 0-1 range

            # Map to our supported languages
            mapped_lang = self.lang_map.get(detected_lang, default)

            if confidence >= self.confidence_threshold:
                return mapped_lang, confidence

        except Exception:
            pass

        return default, 1.0

    def _detect_heuristic(self, text: str, default: str) -> Tuple[str, float]:
        """
        Simple heuristic-based language detection.
        Fast but less accurate - good fallback.
        """
        text_lower = text.lower()

        # First, check for distinctive character sets
        for lang, pattern in self.char_patterns.items():
            if pattern.search(text):
                mapped_lang = self.lang_map.get(lang, default)
                return mapped_lang, 0.9

        # Then check for common words
        words = set(re.findall(r"\b\w+\b", text_lower))
        if not words:
            return default, 1.0

        lang_scores = {}

        for lang, common_words in self.word_patterns.items():
            matches = len(words.intersection(common_words))
            if matches > 0:
                # Score based on percentage of matching words
                score = matches / len(words)
                lang_scores[lang] = score

        if lang_scores:
            # Get the language with highest score
            best_lang = max(lang_scores, key=lang_scores.get)
            confidence = lang_scores[best_lang]

            # Only return if confidence is reasonable
            if confidence >= 0.3:  # Lower threshold for heuristic
                return best_lang, confidence

        return default, 1.0

    def detect_language_simple(self, text: str, default: str = "en") -> str:
        """Simple language detection returning just the language code."""
        lang, _ = self.detect_language(text, default)
        return lang

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return list(set(self.lang_map.values()))


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
        max_models: int = 3,
        eviction_time: int = 300,  # 5 minutes
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


# Enhanced Search Intent Detector with Lazy Loading and Auto Language Detection
class AutoDetectSearchIntentDetector:
    """
    Search intent detector with automatic language detection and lazy model loading.
    """

    def __init__(
        self,
        max_models: int = 3,
        eviction_time: int = 300,
        default_lang: str = "en",
        detection_method: str = "auto",
        confidence_threshold: float = 0.8,
    ):
        self.model_manager = LazySpacyModelManager(max_models=max_models, eviction_time=eviction_time)
        self.default_lang = default_lang

        # Initialize language detector
        self.language_detector = LanguageDetector(method=detection_method, confidence_threshold=confidence_threshold)

        # Performance tracking
        self.request_count = 0
        self.detection_stats = {"auto_detected": 0, "fallback_used": 0, "detection_time_total": 0.0}

    def extract_searchable_content(
        self, text: str, lang: Optional[str] = None, auto_detect: bool = True
    ) -> Tuple[Optional[str], str]:
        """
        Extract searchable content with automatic language detection.

        Args:
            text: Input text to analyze
            lang: Language code (auto-detects if None and auto_detect=True)
            auto_detect: Whether to auto-detect language

        Returns:
            Tuple of (searchable_content, detected_language)
        """
        self.request_count += 1

        # Detect language if not provided
        if lang is None and auto_detect:
            start_time = time.time()
            detected_lang, confidence = self.language_detector.detect_language(text, self.default_lang)
            detection_time = time.time() - start_time

            self.detection_stats["detection_time_total"] += detection_time

            if confidence >= self.language_detector.confidence_threshold:
                self.detection_stats["auto_detected"] += 1
                lang = detected_lang
            else:
                self.detection_stats["fallback_used"] += 1
                lang = self.default_lang
        elif lang is None:
            lang = self.default_lang

        # Get model (lazy loaded)
        nlp = self.model_manager.get_model(lang)

        # Process text using existing logic
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

        # Return best candidate and detected language
        if searchable_chunks:
            best_content = max(set(searchable_chunks), key=len)
            return best_content, lang

        return None, lang

    def should_search(self, text: str, lang: Optional[str] = None, auto_detect: bool = True) -> Tuple[bool, str]:
        """
        Check if text contains searchable content.

        Returns:
            Tuple of (should_search, detected_language)
        """
        content, detected_lang = self.extract_searchable_content(text, lang, auto_detect)
        return content is not None, detected_lang

    def process_search_query(self, text: str, lang: Optional[str] = None) -> Dict:
        """
        Complete search query processing with language detection.

        Returns:
            Dictionary with search decision, content, and metadata
        """
        start_time = time.time()

        searchable_content, detected_lang = self.extract_searchable_content(text, lang)
        should_search = searchable_content is not None

        processing_time = time.time() - start_time

        return {
            "should_search": should_search,
            "searchable_content": searchable_content,
            "detected_language": detected_lang,
            "original_text": text,
            "processing_time_ms": round(processing_time * 1000, 2),
            "model_used": self.model_manager.model_map.get(detected_lang, self.model_manager.default_model),
        }

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
        """Get comprehensive performance and detection statistics."""
        manager_stats = self.model_manager.get_stats()

        avg_detection_time = 0
        if self.detection_stats["auto_detected"] > 0:
            avg_detection_time = (
                self.detection_stats["detection_time_total"]
                / (self.detection_stats["auto_detected"] + self.detection_stats["fallback_used"])
            ) * 1000  # Convert to milliseconds

        return {
            **manager_stats,
            "total_requests": self.request_count,
            "auto_detected": self.detection_stats["auto_detected"],
            "fallback_used": self.detection_stats["fallback_used"],
            "detection_accuracy": (
                self.detection_stats["auto_detected"]
                / max(1, self.detection_stats["auto_detected"] + self.detection_stats["fallback_used"])
            )
            * 100,
            "avg_detection_time_ms": round(avg_detection_time, 2),
            "supported_languages": self.language_detector.get_supported_languages(),
        }
