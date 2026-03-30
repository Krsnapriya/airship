"""
FastAPI server wrapping Airship.
Exposes OpenEnv-compliant endpoints: /reset, /step, /state, /grader, /baseline, /health
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Action  # pyre-ignore
from server.env import AirshipEnv  # pyre-ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.envs = {}
    yield


app = FastAPI(
    title="Airship API",
    description="Realistic eXecution benchmark for debugging agents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_env() -> AirshipEnv:
    env = app.state.envs.get("current")
    if env is None:
        raise HTTPException(status_code=400, detail="No active environment. Call /reset first.")
    return env


VALID_DIFFICULTIES = {"easy", "medium", "hard", "extreme"}
VALID_SPLITS = {"train", "test", "ood"}


@app.post("/reset")
def reset(
    difficulty: str = Query("easy", description="easy, medium, hard, extreme"),
    split: str = Query("test", description="train, test, ood"),
    chaos_mode: bool = Query(False, description="Enable live adversarial Meta-Controller"),
    seed: int = Query(42),
):
    if difficulty not in ["easy", "medium", "hard", "extreme"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty")
    if split not in ["train", "test", "ood"]:
        raise HTTPException(status_code=400, detail="Invalid split")

    try:
        env = AirshipEnv(seed=seed)
        app.state.envs["current"] = env
        obs = env.reset(
            difficulty=difficulty,
            split=split,
            chaos_mode=chaos_mode,
            seed=seed
        )
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(action: Action):
    env = _get_env()
    try:
        obs, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    env = _get_env()
    return env.state.model_dump()


@app.get("/grader")
def grader():
    env = _get_env()
    score = env.grade_trajectory(env.trajectory)
    return {
        "score": score.final(),
        "breakdown": score.model_dump()
    }


@app.get("/baseline")
def baseline():
    """Run baseline evaluation. Returns scores or error on failure."""
    try:
        from baseline import run_baseline  # pyre-ignore
        return run_baseline()
    except Exception as e:
        return {
            "error": str(e),
            "note": "Baseline failed safely – reproducible on local run via 'python baseline.py'"
        }


@app.get("/health")
def health():
    return {"status": "ok", "environment": "Airship", "version": "1.0.0"}


@app.get("/")
def root():
    return {"message": "Airship API is running. Submit POST requests to /reset and /step."}
