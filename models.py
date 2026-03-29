from pydantic import BaseModel, Field  # pyre-ignore
from typing import List, Optional, Dict


class Action(BaseModel):
    type: str = Field(...)
    target: Optional[str] = Field(None)
    content: Optional[str] = Field(None)


class Observation(BaseModel):
    visible_files: List[str] = Field(default_factory=list)
    logs: str = Field("")
    test_results: Optional[str] = Field(None)
    time_remaining: int = Field(0)


class HiddenState(BaseModel):
    true_bug_locations: List[str] = Field(default_factory=list)
    bug_type: str = Field(...)
    eta: float = Field(0.0)
    delta: float = Field(0.0)
    # === META-CONTROLLER LIVE FIELDS (added now) ===
    current_eta: float = Field(0.0)
    current_delta: float = Field(0.0)
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)


class ObservableState(BaseModel):
    files: Dict[str, str] = Field(default_factory=dict)
    original_files: Dict[str, str] = Field(default_factory=dict)
    bug_location: str = Field(...)
    difficulty: str = Field(...)
    split: str = Field(...)
    steps_taken: int = Field(0)
    max_steps: int = Field(...)
    resolved: bool = Field(False)
    
    files_opened: List[str] = Field(default_factory=list)
    edits_made: List[Dict] = Field(default_factory=list)
    tests_run: int = Field(0)
    logs_analyzed: int = Field(0)


class Score(BaseModel):
    correctness: float = Field(0.0, ge=0.0, le=1.0)
    efficiency: float = Field(0.0, ge=0.0, le=1.0)
    reasoning_quality: float = Field(0.0, ge=0.0, le=1.0)
    robustness: float = Field(0.0, ge=0.0, le=1.0)

    def final(self) -> float:
        score = (
            0.4 * self.correctness +
            0.2 * self.efficiency +
            0.2 * self.reasoning_quality +
            0.2 * self.robustness
        )
        return min(max(score, 0.0), 1.0)
