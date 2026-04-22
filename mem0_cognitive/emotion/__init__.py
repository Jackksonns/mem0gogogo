"""
Emotion Analysis Module for Mem0-Cognitive

Provides zero-shot LLM-based emotion extraction with lexicon fallback
for computing emotional salience scores in memory retention.
"""

from mem0_cognitive.emotion.analyzer import EmotionAnalyzer
from mem0_cognitive.emotion.configs import EmotionConfig

__all__ = ["EmotionAnalyzer", "EmotionConfig"]
