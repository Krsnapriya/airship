import random
import numpy as np
import os
from server.env import AirshipEnv  # pyre-ignore
from run_baseline import HeuristicAgent  # pyre-ignore
from models import Action, Score  # pyre-ignore

def run_baseline(difficulty: str = "easy", num_seeds: int = 5) -> dict:
    """
    Standardized evaluation wrapper for the benchmark.
    Returns scores across difficulties.
    """
    tasks = ["easy", "medium", "hard", "extreme"]
    results = {}
    
    # Simple deterministic seed range as per README
    seeds = [40, 41, 42, 43, 44]
    
    print(f"Running Baseline Evaluation (Heuristic Agent)...")
    
    for task_id in tasks:
        task_scores = []
        for seed in seeds:
            env = AirshipEnv(seed=seed)
            obs = env.reset(difficulty=task_id, split="test", chaos_mode=False, seed=seed)
            
            agent = HeuristicAgent()
            trajectory = []
            done = False
            step_count = 0
            
            while not done and step_count < 30:
                step_count += 1
                action = agent(obs)
                obs, reward, done, info = env.step(action)
                trajectory.append(action)
                
            final_score: Score = env.grade_trajectory(trajectory)
            task_scores.append(float(final_score.final()))
            
        mean = float(np.mean(task_scores))
        std = float(np.std(task_scores))
        results[task_id] = {"mean": round(mean, 3), "std": round(std, 3)}
        print(f"Task: {task_id} -> {mean:.3f} ± {std:.3f}")
        
    return results

if __name__ == "__main__":
    scores = run_baseline()
    print("\nFinal Results:", scores)
