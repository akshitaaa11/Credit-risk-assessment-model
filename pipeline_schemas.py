from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class PipelineRequest:
    csv_path: str
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    smote_ratio: float = 1.0
    model_dir: str = 'models'
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResponse:
    status: str
    model_metrics: Optional[Dict[str, Dict[str, float]]] = None
    predictions: Optional[Dict[str, Any]] = None
    best_model_name: Optional[str] = None
    error: Optional[str] = None
