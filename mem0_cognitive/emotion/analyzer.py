"""
Emotion Analyzer: Zero-shot LLM-based emotion extraction with lexicon fallback

This module implements the emotion analysis component described in Section 3.1 of
our ACL 2026 paper. It extracts emotional intensity, valence, and arousal from
user utterances using zero-shot LLM prompting, with a lexicon-based fallback for
robustness.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, Tuple

from mem0_cognitive.emotion.configs import EmotionConfig

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """
    Extracts emotional salience from text using zero-shot LLM prompting.
    
    As described in our ACL 2026 paper (Section 3.1), this analyzer computes
    emotional intensity E ∈ [0, 1] which modulates the Affective Retention Score:
    
        S_eff = S_base · (1 + λ·E)
    
    where λ ∈ [0, 2] is the emotional inertia coefficient.
    
    Example usage (as shown in paper Appendix):
        >>> analyzer = EmotionAnalyzer()
        >>> result = analyzer.extract("I can't believe you remembered my birthday!")
        >>> print(result)
        {'intensity': 0.85, 'valence': 'positive', 'arousal': 'high', ...}
    """
    
    # Prompt template as provided in paper Appendix (with proper formatting)
    PROMPT_TEMPLATE = """You are an emotion analyzer. Given the following user utterance, rate its emotional intensity on a scale of {min_scale} (neutral) to {max_scale} (highly charged). Consider valence and arousal.

Utterance: "{utterance}"

Output JSON with the following structure:
{{
    "intensity": <float between {min_scale} and {max_scale}>,
    "valence": "<positive|negative|neutral>",
    "arousal": "<low|moderate|high>",
    "rationale": "<brief explanation>"
}}

Provide only the JSON output, no additional text."""

    def __init__(self, config: Optional[EmotionConfig] = None):
        """
        Initialize the Emotion Analyzer.
        
        Args:
            config: EmotionConfig object with model and parameter settings.
                   If None, uses default configuration.
        """
        self.config = config or EmotionConfig()
        self._llm_client = None
        
        # Fix random seed for reproducibility (ACL 2026 experiments)
        if self.config.seed is not None:
            import random
            import numpy as np
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            
        logger.info(f"EmotionAnalyzer initialized with model={self.config.model_name}, "
                   f"temperature={self.config.temperature}, seed={self.config.seed}")
    
    def extract(self, utterance: str, scale: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Extract emotional features from a single utterance.
        
        Args:
            utterance: The text to analyze
            scale: Optional override for (min, max) intensity scale
            
        Returns:
            Dictionary containing:
                - intensity: Float in [0, 1] representing emotional strength
                - valence: One of 'positive', 'negative', 'neutral'
                - arousal: One of 'low', 'moderate', 'high'
                - rationale: Brief explanation from LLM
                - method: 'llm' or 'lexicon' indicating extraction method
                
        Example from paper Appendix:
            Input: "I can't believe you remembered my birthday! That means so much to me."
            Output: {"intensity": 0.85, "valence": "positive", 
                     "rationale": "Expression of surprise and gratitude..."}
        """
        scale = scale or self.config.scale
        
        try:
            # Primary method: Zero-shot LLM extraction
            result = self._extract_via_llm(utterance, scale)
            result['method'] = 'llm'
            logger.debug(f"Emotion extracted via LLM: {result['intensity']}")
            return result
            
        except Exception as e:
            logger.warning(f"LLM emotion extraction failed: {e}. Falling back to lexicon.")
            
            # Fallback: Lexicon-based extraction
            if self.config.enable_lexicon_fallback:
                result = self._extract_via_lexicon(utterance, scale)
                result['method'] = 'lexicon'
                return result
            else:
                # Return neutral if fallback disabled
                return {
                    'intensity': 0.0,
                    'valence': 'neutral',
                    'arousal': 'low',
                    'rationale': 'Emotion extraction disabled',
                    'method': 'none'
                }
    
    def _extract_via_llm(self, utterance: str, scale: Tuple[float, float]) -> Dict[str, Any]:
        """Extract emotion using zero-shot LLM prompting."""
        # Lazy initialization of LLM client
        if self._llm_client is None:
            self._init_llm_client()
        
        # Format prompt with scale parameters
        prompt = self.PROMPT_TEMPLATE.format(
            utterance=utterance,
            min_scale=scale[0],
            max_scale=scale[1]
        )
        
        # Call LLM with temperature=0 for reproducibility
        # Note: As discussed with advisor, we acknowledge minor API non-determinism
        # even at temperature=0 (see paper Limitations section)
        response = self._llm_client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are an emotion analysis assistant. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature,
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        response_text = response.choices[0].message.content.strip()
        result = json.loads(response_text)
        
        # Validate and normalize intensity to [0, 1]
        intensity = float(result.get('intensity', 0.0))
        intensity_normalized = (intensity - scale[0]) / (scale[1] - scale[0])
        intensity_normalized = max(0.0, min(1.0, intensity_normalized))
        
        return {
            'intensity': intensity_normalized,
            'valence': result.get('valence', 'neutral'),
            'arousal': result.get('arousal', 'low'),
            'rationale': result.get('rationale', '')
        }
    
    def _extract_via_lexicon(self, utterance: str, scale: Tuple[float, float]) -> Dict[str, Any]:
        """
        Fallback lexicon-based emotion extraction.
        
        Uses simple keyword matching when LLM is unavailable.
        This is less accurate but ensures robustness.
        """
        # Simplified lexicon (in production, use NRC or VADER)
        positive_words = {
            'love', 'amazing', 'wonderful', 'fantastic', 'great', 'excellent',
            'happy', 'joy', 'excited', 'thrilled', 'delighted', 'grateful',
            'thank', 'means', 'much', 'believe', 'remember'
        }
        negative_words = {
            'hate', 'terrible', 'awful', 'horrible', 'bad', 'worst',
            'sad', 'angry', 'frustrated', 'disappointed', 'upset'
        }
        intensifiers = {
            'very', 'extremely', 'absolutely', 'totally', 'really', 'so'
        }
        
        words = set(re.findall(r'\b\w+\b', utterance.lower()))
        
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        int_count = len(words & intensifiers)
        
        # Compute intensity with intensifier boost
        base_intensity = abs(pos_count - neg_count)
        intensity = min(1.0, base_intensity * (1 + 0.2 * int_count) / 5.0)
        
        # Determine valence
        if pos_count > neg_count:
            valence = 'positive'
        elif neg_count > pos_count:
            valence = 'negative'
        else:
            valence = 'neutral'
        
        # Simple arousal heuristic
        arousal = 'high' if intensity > 0.6 else ('moderate' if intensity > 0.3 else 'low')
        
        return {
            'intensity': intensity,
            'valence': valence,
            'arousal': arousal,
            'rationale': f'Lexicon match: {pos_count} positive, {neg_count} negative keywords'
        }
    
    def _init_llm_client(self):
        """Initialize LLM client (OpenAI-compatible API)."""
        try:
            from openai import OpenAI
            self._llm_client = OpenAI()
        except ImportError:
            raise ImportError(
                "OpenAI package required for emotion extraction. "
                "Install with: pip install openai"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
