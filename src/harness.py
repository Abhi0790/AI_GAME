"""Build and run one game from a GameConfig. The only place agents are
constructed.

The board and the knobs are process-global for the length of a game, so a
runner must be built and stepped inside `game_setup`.
"""
import multiprocessing
import os
import random
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from src.common.config import GameConfig
from src.agents import agent as agent_module
from src.agents.agent import Agent, HumanAgent
from src.agents.chaos import ChaosAgent
from src.agents.negotiation import strategy as strategy_module
from src.agents.negotiation.personas import PERSONAS
from src.agents.planner import planner as planner_module
from src.agents.planner.planner import PlannerConfig
from src.agents.trust import model as trust_model_module, rules as trust_rules_module
from src.engine.board import use_board
from src.engine import runner as runner_module
from src.engine.runner import GameRunner

# Modules whose UPPERCASE globals are tuning knobs. `apply_knobs` looks a name
# up across these, so a new constant is overridable the moment it is defined.
KNOB_MODULES = [trust_model_module, trust_rules_module, planner_module,
                strategy_module, agent_module, runner_module]


def knob_values() -> Dict[str, float]:
    """Every knob and its current value — what a run should print."""
    out: Dict[str, float] = {}
    for m in KNOB_MODULES:
        for name, v in vars(m).items():
            if name.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool):
                out[name] = v
    return out


@contextmanager
def apply_knobs(knobs: Dict[str, float]) -> Iterator[None]:
    """Rebind module constants for the block, then put them back."""
    saved = []
    for name, value in knobs.items():
        owners = [m for m in KNOB_MODULES if name in vars(m)]
        if not owners:
            raise KeyError(f"unknown knob {name!r}; known: {sorted(knob_values())}")
        for m in owners:
            saved.append((m, name, getattr(m, name)))
            setattr(m, name, value)
    try:
        yield
    finally:
        for m, name, old in reversed(saved):
            setattr(m, name, old)


def build_agents(cfg: GameConfig) -> List:
    agents = []
    for player, persona in cfg.seats().items():
        if player == cfg.human_seat:
            agents.append(HumanAgent(player, persona))
        elif player == cfg.chaos_seat:
            agents.append(ChaosAgent(player, mode=cfg.chaos_mode, seed=cfg.seed))
        else:
            kw = {"search": cfg.search, "node_budget": cfg.node_budget, "depth": cfg.depth,
                  **cfg.seat_planner.get(player, {})}
            a = Agent(player, persona, PlannerConfig(**kw))
            a.planner.config.reputation_cost_coefficient = (
                PERSONAS[persona].reputation_cost * cfg.rep_cost_scale)
            agents.append(a)
    return agents


# The board and the knobs are process-global while a game runs, so two games
# with different setups must not be inside game_setup at once. Re-entrant: a
# script that wraps play() in its own game_setup is nesting the same config.
_SETUP_LOCK = threading.RLock()


@contextmanager
def game_setup(cfg: GameConfig) -> Iterator[None]:
    """Install this config's board and knobs for the block."""
    with _SETUP_LOCK, use_board(cfg.make_board()), apply_knobs(cfg.knobs):
        yield


@contextmanager
def read_only() -> Iterator[None]:
    """Do hypothetical work without moving the stream the real game draws from.

    The planner samples opponent worlds while it searches, so anything that
    re-derives a decision — the dashboard's detailed mode, say — would
    otherwise change what happens next.
    """
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


def build_runner(cfg: GameConfig, **runner_kwargs) -> GameRunner:
    """Seed, build agents, build the runner. Call inside `game_setup(cfg)`."""
    random.seed(cfg.seed)
    runner = GameRunner(build_agents(cfg), **runner_kwargs)
    runner.negotiation_rounds = cfg.negotiation_rounds
    runner.config = cfg
    return runner


def play(cfg: GameConfig, verbose: bool = False) -> GameRunner:
    """One complete game from a config. Deterministic in cfg.seed."""
    with game_setup(cfg):
        runner = build_runner(cfg)
        runner.run(verbose=verbose)
    return runner


# A game peaks near 50 MB of RSS; the allowance leaves ~10x headroom for larger boards.
PER_WORKER_BYTES = 512 * 2 ** 20
# Games per worker before its pool is replaced, so a slow leak cannot accumulate.
TASKS_PER_WORKER = 25


def _available_memory() -> int:
    """Bytes free for new work right now: MemAvailable on Linux, the kernel's
    free percentage on macOS, and never more than a container's remaining limit."""
    total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    available = total
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
    except OSError:
        try:
            level = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"],
                                   capture_output=True, text=True, timeout=5).stdout
            available = total * int(level) // 100
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    for limit_path, usage_path in (
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
             "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            with open(limit_path) as f, open(usage_path) as g:
                limit, usage = f.read().strip(), g.read().strip()
        except OSError:
            continue
        if limit.isdigit() and usage.isdigit():
            available = min(available, int(limit) - int(usage))
        break
    return max(0, available)


def default_workers() -> int:
    """GAME_WORKERS if set, else one per spare core, capped so the workers fit
    in half the memory available right now."""
    if os.environ.get("GAME_WORKERS"):
        return max(1, int(os.environ["GAME_WORKERS"]))
    by_memory = _available_memory() // 2 // PER_WORKER_BYTES
    return max(1, min((os.cpu_count() or 2) - 1, by_memory))


def _play_one(job):
    cfg, reduce = job
    runner = play(cfg)
    if reduce is None:
        return runner
    with game_setup(cfg):
        return reduce(cfg, runner)


def play_many(cfgs: Iterable[GameConfig],
              reduce: Optional[Callable[[GameConfig, GameRunner], Any]] = None,
              workers: Optional[int] = None) -> List[Any]:
    """`play` over many configs in worker processes, results in input order.

    `reduce(cfg, runner)` runs in the worker, inside the game's setup, so only
    its result is sent back; without it whole runners are returned. It must be
    a module-level function.
    """
    jobs = [(cfg, reduce) for cfg in cfgs]
    workers = min(workers or default_workers(), len(jobs))
    # Spawned workers re-import __main__, which a REPL, stdin or `-c` does not have.
    main_file = getattr(sys.modules["__main__"], "__file__", None)
    if workers <= 1 or not (main_file and os.path.exists(main_file)):
        return [_play_one(job) for job in jobs]
    # A fresh pool per batch bounds any per-process growth. Not max_tasks_per_child:
    # on 3.13.5 retired workers were not replaced and the pool hung with work pending.
    results: List[Any] = []
    batch = workers * TASKS_PER_WORKER
    for start in range(0, len(jobs), batch):
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results.extend(pool.map(_play_one, jobs[start:start + batch]))
    return results
