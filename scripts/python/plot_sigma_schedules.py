"""Plot sigma(t) from CoordScoreDiffusion for VE and VP schedules.

Reads schedule + temperature settings from a training YAML and produces a PNG.

Example:
  /path/to/python scripts/python/plot_sigma_schedules.py \
    --config configs/training/energy_force_om25.yml \
    --out runs/sigma_schedules.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from src.model.flows import CoordScoreDiffusion, SDEParams


def _get_float(dct: dict, key: str, default: float) -> float:
    val = dct.get(key, default)
    if val is None:
        return float(default)
    return float(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML")
    parser.add_argument("--out", type=str, default="runs/sigma_schedules.png", help="Output PNG path")
    parser.add_argument("--n", type=int, default=200, help="Number of t points")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    sim = cfg.get("simulation_params", {}) or {}

    # Values as used by CoordScoreDiffusion in this repo.
    sigma_min = _get_float(sim, "sigma_min", 0.01)
    sigma_max = _get_float(sim, "sigma_max", 50.0)
    beta_min = _get_float(sim, "beta_min", 0.1)
    beta_max = _get_float(sim, "beta_max", 20.0)
    vp_sigma_scale = _get_float(sim, "vp_sigma_scale", 1.0)

    temperature = _get_float(sim, "temperature", 1.0)
    temperature_min = _get_float(sim, "temperature_min", temperature)
    temperature_max = _get_float(sim, "temperature_max", temperature)

    # Pick two temperatures to visualize.
    temps = [temperature_min, temperature_max]
    temps = [float(temps[0]), float(temps[1])]

    # Time grid in flow-matching convention: t=0 noisy -> t=1 clean
    t = torch.linspace(0.0, 1.0, int(args.n)).unsqueeze(-1)

    ve = CoordScoreDiffusion(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        dim=3,
        sde=SDEParams(kind="ve", sigma_min=sigma_min, sigma_max=sigma_max),
    )
    vp = CoordScoreDiffusion(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        dim=3,
        sde=SDEParams(
            kind="vp",
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            beta_min=beta_min,
            beta_max=beta_max,
            vp_sigma_scale=vp_sigma_scale,
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    t_np = t.squeeze(-1).cpu().numpy()

    def plot_schedule(ax, schedule_name: str, diffusion: CoordScoreDiffusion) -> None:
        ax2 = ax.twinx()

        for T in temps:
            sig = diffusion.sigma(t, temperature=T).squeeze(-1).detach().cpu().numpy()
            a = diffusion.alpha(t, temperature=T).squeeze(-1).detach().cpu().numpy()

            ax.plot(t_np, sig, label=f"σ, T={T:g}")
            ax2.plot(t_np, a, linestyle="--", label=f"α, T={T:g}")

        ax.set_title(f"{schedule_name}: σ(t) and α(t)")
        ax.set_xlabel("t (0=noisy → 1=clean)")
        ax.set_ylabel("sigma")
        ax2.set_ylabel("alpha")
        ax.grid(True, alpha=0.3)

        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    plot_schedule(axes[0], "VE", ve)
    plot_schedule(axes[1], "VP", vp)

    fig.suptitle(
        "CoordScoreDiffusion schedules from YAML\n"
        f"VE(sigma_min={sigma_min:g}, sigma_max={sigma_max:g}); "
        f"VP(beta_min={beta_min:g}, beta_max={beta_max:g}, vp_sigma_scale={vp_sigma_scale:g})"
    )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)

    print(f"Wrote: {out_path.resolve()}")
    print(f"Temperatures shown: {temps[0]} and {temps[1]}")


if __name__ == "__main__":
    main()
