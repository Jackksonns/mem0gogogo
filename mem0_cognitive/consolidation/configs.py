"""
Configuration for Sleep Consolidation Engine
"""

from dataclasses import dataclass


@dataclass
class ConsolidationConfig:
    """
    Configuration parameters for sleep-based memory consolidation.
    
    As described in paper Section 3.3, the consolidation engine performs offline
    memory reconsolidation inspired by hippocampus-to-neocortex transfer in human
    sleep. It clusters similar memories and generalizes specifics into abstract
    knowledge.
    
    Attributes:
        enable_consolidation: Whether to run consolidation cycles
        consolidation_interval_hours: Hours between consolidation cycles (default: 6)
        min_memories_for_consolidation: Minimum number of memories required to trigger
        clustering_threshold: Similarity threshold for clustering memories (0-1)
        generalization_strategy: Strategy for merging similar memories
            - 'average': Average embeddings and content
            - 'summarize': Use LLM to generate summary
            - 'keep_best': Keep highest-scoring memory
        max_cluster_size: Maximum memories per cluster before forced summarization
        retention_score_cutoff: Only consolidate memories with score > cutoff
    """
    enable_consolidation: bool = True
    consolidation_interval_hours: int = 6
    min_memories_for_consolidation: int = 10
    clustering_threshold: float = 0.85
    generalization_strategy: str = "summarize"  # 'average', 'summarize', 'keep_best'
    max_cluster_size: int = 20
    retention_score_cutoff: float = 0.2
    
    # LLM settings for summarization
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3  # Slight creativity for summarization
    
    # Logging and monitoring
    verbose: bool = False
    
    def __post_init__(self):
        if not 0 <= self.clustering_threshold <= 1:
            raise ValueError("Clustering threshold must be in [0, 1]")
        if self.consolidation_interval_hours < 1:
            raise ValueError("Consolidation interval must be at least 1 hour")
        if self.min_memories_for_consolidation < 1:
            raise ValueError("Minimum memories must be at least 1")
        if self.generalization_strategy not in ['average', 'summarize', 'keep_best']:
            raise ValueError(
                f"Invalid strategy: {self.generalization_strategy}. "
                "Must be 'average', 'summarize', or 'keep_best'"
            )
