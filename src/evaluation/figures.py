"""The six report figures.

One function per figure, each taking already-collected data and returning a
matplotlib Figure. Nothing here runs a game — `scripts/make_figures.py` does
the experiments and hands the results over, so a figure can be redrawn
without replaying anything.

    1. betrayal rate vs turn, per persona
    2. Vcoop at the moment of a break, vs turn
    3. rating by persona x search horizon
    4. turns to coalition
    5. reliability diagram (with the Brier score)
    6. win rate vs adjudications per turn, expectiminimax vs MCTS
"""

from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import MIN_GAMES

# One colour per persona everywhere, so a reader can carry the legend between
# figures without re-reading it.
PERSONA_COLOURS = {
    "Honest": "#4c9f70",
    "Opportunist": "#c1666b",
    "Vengeful": "#d4a04c",
    "Paranoid": "#5b7db1",
    "Chaos": "#8a8a8a",
}
GRID = dict(alpha=0.25, linewidth=0.6)


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, **GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def betrayal_rate_vs_turn(series: Dict[str, List[Optional[float]]]):
    """Figure 1. series: persona -> per-turn betrayal rate (None = no deals)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for persona, values in sorted(series.items()):
        turns = [i + 1 for i, v in enumerate(values) if v is not None]
        rates = [v for v in values if v is not None]
        if not turns:
            continue
        ax.plot(turns, rates, marker="o", markersize=4, linewidth=1.6,
                label=persona, color=PERSONA_COLOURS.get(persona))
    _style(ax, "Betrayal rate by turn, per persona", "Turn", "Commitments broken / held")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def vcoop_at_break(rows: List[Dict[str, Any]]):
    """Figure 2. One point per broken deal: what the partner was still worth.

    The claim the figure has to support is that betrayal is not scripted —
    breaks cluster where Vcoop has already collapsed, or where the horizon
    term has.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for persona in sorted({r["persona"] for r in rows}):
        pts = [r for r in rows if r["persona"] == persona]
        ax.scatter([r["turn"] for r in pts], [r["vcoop"] for r in pts],
                   s=28, alpha=0.75, label=persona,
                   color=PERSONA_COLOURS.get(persona), edgecolors="none")
    _style(ax, "Vcoop(partner) at the moment the promise was broken",
           "Turn", "Vcoop, in centres")
    if rows:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def rating_by_persona_and_horizon(data: Dict[int, Dict[str, float]]):
    """Figure 3. data: search depth -> persona -> mean final centres."""
    fig, ax = plt.subplots(figsize=(7, 4))
    horizons = sorted(data)
    personas = sorted({p for d in data.values() for p in d})
    width = 0.8 / max(1, len(personas))

    for i, persona in enumerate(personas):
        xs = [h + (i - (len(personas) - 1) / 2) * width for h in range(len(horizons))]
        ys = [data[h].get(persona, 0.0) for h in horizons]
        ax.bar(xs, ys, width=width, label=persona, color=PERSONA_COLOURS.get(persona))

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"depth {h}" for h in horizons])
    _style(ax, "Rating by persona and search horizon", "Search horizon",
           "Mean final centres")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def turns_to_coalition(turns: List[Optional[int]]):
    """Figure 4. How long until an alliance forms."""
    fig, ax = plt.subplots(figsize=(7, 4))
    formed = [t for t in turns if t is not None]

    if formed:
        parts = ax.boxplot([formed], tick_labels=[f"n={len(formed)} of {len(turns)}"],
                           patch_artist=True, widths=0.5)
        parts["boxes"][0].set_facecolor("#4c9f70")
        parts["boxes"][0].set_alpha(0.6)
        ax.scatter([1] * len(formed), formed, s=18, color="#222", zorder=3, alpha=0.6)
    _style(ax, "Turns until the first standing alliance", "", "Turn")
    fig.tight_layout()
    return fig


def reliability_diagram(bins: List[Dict[str, Any]], brier: Optional[float],
                        baseline: Optional[float] = None):
    """Figure 5. The only figure that tests whether the trust numbers mean
    anything: predicted P(keeps) against the fraction actually kept."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#888",
            label="perfect calibration")

    if bins:
        xs = [b["mean_predicted"] for b in bins]
        ys = [b["observed_rate"] for b in bins]
        sizes = [30 + 6 * b["count"] for b in bins]
        ax.plot(xs, ys, linewidth=1.6, color="#5b7db1", zorder=2)
        ax.scatter(xs, ys, s=sizes, color="#5b7db1", zorder=3,
                   edgecolors="white", linewidth=0.8, label="observed (area = n)")

    title = "Reliability of P(keeps this commitment)"
    if brier is not None:
        title += f"\nBrier {brier:.3f}"
        if baseline is not None:
            title += f"  (base rate {baseline:.3f})"
    _style(ax, title, "Predicted P(keeps)", "Fraction actually kept")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def win_rate_vs_budget(rows: List[Dict[str, Any]]):
    """Figure 6. rows: {search, adjudications, win_rate, win_rate_lo/hi, games}.

    The comparison the report asks for is at *equal budget*, which is why
    both searches count adjudications through the same counter. Each point
    carries its 95% interval: at two games per cell the old version of this
    figure plotted only 0.0 and 1.0 and looked decisive.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    colours = {"expectiminimax": "#5b7db1", "mcts": "#c1666b"}
    for search in sorted({r["search"] for r in rows}):
        pts = sorted((r for r in rows if r["search"] == search),
                     key=lambda r: r["adjudications"])
        # Asymmetric Wilson bars: a cell at 0 wins has no room below it, and
        # a symmetric bar there drew the eye to an interval that cannot exist.
        lo = [p["win_rate"] - p.get("win_rate_lo", p["win_rate"]) for p in pts]
        hi = [p.get("win_rate_hi", p["win_rate"]) - p["win_rate"] for p in pts]
        ax.errorbar([p["adjudications"] for p in pts], [p["win_rate"] for p in pts],
                    yerr=[lo, hi],
                    marker="o", markersize=5, linewidth=1.6, capsize=3,
                    label=search, color=colours.get(search))

    n = min((r.get("games", 0) for r in rows), default=0)
    title = f"Win rate against adjudications per turn (n={n} games per cell)"
    _style(ax, title, "Adjudications per turn (search budget)", "Win rate")
    if rows and n < MIN_GAMES:
        ax.text(0.5, 0.97, f"n below threshold (MIN_GAMES={MIN_GAMES})",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color="#c1666b")
    ax.set_ylim(-0.05, 1.05)
    if rows:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig
