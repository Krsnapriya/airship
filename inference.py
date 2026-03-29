import os
import json
import re
import random
import sys
import numpy as np  # pyre-ignore
from typing import Optional, List, Dict, Any

from openai import OpenAI  # pyre-ignore
from models import Action, Score, Observation  # pyre-ignore
from server.env import AirshipEnv  # pyre-ignore

# Standardized configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("Error: HF_TOKEN or OPENAI_API_KEY environment variable is required.")
    sys.exit(1)

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

TEMPERATURE = 0


def safe_parse(text: str) -> Optional[Dict[str, Any]]:
    """Safe extraction of JSON from model response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def llm_agent_step(
    obs_dict: Dict[str, Any],
    step_num: int,
    max_steps: int,
    available_files: List[str],
) -> Action:
    """Step logic for the LLM agent."""
    
    prompt = f"""You are a code review and debugging agent.
You must decide exactly one action per step.

Current state:
- Step {step_num} of {max_steps}
- Visible files: {obs_dict.get('visible_files', [])}
- Logs: {obs_dict.get('logs', 'No logs yet')}
- Test results: {obs_dict.get('test_results', 'No tests run yet')}
- Time remaining: {obs_dict.get('time_remaining', 0)} steps
- Available files in repo: {available_files}

Available actions:
1. {{"type": "analyze_logs"}} 
2. {{"type": "open_file", "target": "<filename>"}}
3. {{"type": "edit_file", "target": "<filename>", "content": "<new content>"}}
4. {{"type": "run_tests"}}

Return only valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        text = response.choices[0].message.content or ""
        parsed = safe_parse(text)
    except Exception as e:
        print(f"API Call failure: {e}")
        parsed = None

    if parsed and "type" in parsed:
        action_type = str(parsed["type"])
        target = parsed.get("target")
        content = parsed.get("content")

        valid_types = {"analyze_logs", "open_file", "edit_file", "run_tests"}
        if action_type not in valid_types:
            return Action(type="analyze_logs")

        if action_type in ("open_file", "edit_file") and target:
            target = str(target)
            if target not in available_files:
                target = available_files[0] if available_files else "unknown.py"

        return Action(
            type=action_type,
            target=str(target) if target else None,
            content=str(content) if content else None,
        )

    return Action(type="analyze_logs") if step_num <= 1 else Action(type="run_tests")


def run_inference():
    """Executes evaluation across all tasks."""
    tasks = ["easy", "medium", "hard", "extreme"]
    results = {}

    print(f"Starting inference with model: {MODEL_NAME}")
    print(f"Endpoint: {API_BASE_URL}")

    for task_id in tasks:
        print(f"Task: {task_id}")
        
        env = AirshipEnv(seed=42)
        obs = env.reset(difficulty=task_id, split="test", seed=42)

        done = False
        step_count = 0
        trajectory = []
        available_files = list(env.state.files.keys()) if env.state else []

        while not done and step_count < 30:
            step_count += 1
            obs_dict = obs.model_dump()
            action = llm_agent_step(obs_dict, step_count, 30, available_files)
            
            try:
                obs, reward, done, info = env.step(action)
                trajectory.append(action)
            except Exception:
                continue

        final_score: Score = env.grade_trajectory(trajectory)
        score_val = float(final_score.final())
        results[task_id] = score_val
        print(f"Score: {score_val:.3f}")

    return results


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    try:
        scores = run_inference()
        print("\nFinal Results:")
        for task, score in scores.items():
            print(f"{task}: {score:.3f}")
    except Exception as e:
        print(f"Execution failure: {e}")
        sys.exit(1)
