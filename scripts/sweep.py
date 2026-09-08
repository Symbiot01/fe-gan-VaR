"""Run a sweep of training runs with multiple seeds and configurations.

Supports parallel execution using torch.multiprocessing for efficient A100 usage.

Usage:
    # Single config sweep
    python -m scripts.sweep --config configs/fegan_hist_wgan.yaml --seeds 0,1,2,3,4

    # Full sweep from plan
    python -m scripts.sweep --sweep configs/sweep_plan_a.yaml --workers 10

    # Dry run (2 seeds, 5 epochs)
    python -m scripts.sweep --config configs/fegan_hist_wgan.yaml --seeds 0,1 --epochs 5
"""

from __future__ import annotations

import argparse
import csv
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from fe_gan.data import load_vix_dataset
from fe_gan.trainer import TrainerConfig, train_model
from fe_gan.utils import get_system_info, save_json


@dataclass
class SweepTask:
    """A single sweep task (config + seed)."""

    config_name: str
    config_path: Path
    seed: int
    epochs: int
    out_dir: Path
    use_amp: bool = False
    cudnn_benchmark: bool = False


def run_single_task(
    task: SweepTask,
    device: torch.device,
    train_data,
    eval_data,
) -> dict[str, Any]:
    """Run a single training task.

    Args:
        task: The sweep task to run.
        device: Device to use.
        train_data: Training dataset.
        eval_data: Evaluation dataset.

    Returns:
        Evaluation results dictionary.
    """
    # Load config
    cfg = OmegaConf.load(task.config_path)

    # Apply optimizations
    if task.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # Create TrainerConfig
    config = TrainerConfig(
        model_type=cfg.get("model_type", "wgan"),
        loss_type=cfg.get("loss_type", "wgan"),
        noise_dim=cfg.get("noise_dim", 100),
        gen_depth=cfg.get("gen_depth", 10),
        gen_width=cfg.get("gen_width", 1000),
        critic_depth=cfg.get("critic_depth", 5),
        critic_width=cfg.get("critic_width", 100),
        hist_dim=cfg.get("hist_dim", 250),
        preproc_hidden=cfg.get("preproc_hidden", 512),
        preproc_feat_dim=cfg.get("preproc_feat_dim", 250),
        epochs=task.epochs,
        batch_size=cfg.get("batch_size", 100),
        n_critic=cfg.get("n_critic", 5),
        clip_value=cfg.get("clip_value", 0.01),
        lr=cfg.get("lr", 5e-5),
        alpha=cfg.get("alpha", 0.05),
        tailgan_wgan_weight=cfg.get("tailgan_wgan_weight", 0.5),
        tailgan_fz_loss_clip_value=cfg.get("tailgan_fz_loss_clip_value"),
        generator_grad_clip_norm=cfg.get("generator_grad_clip_norm"),
        eval_every=cfg.get("eval_every", 10),
        eval_samples=cfg.get("eval_samples", 1000),
        seed=task.seed,
        out_dir=str(task.out_dir),
        save_checkpoint=False,  # Never save checkpoints in sweep
    )

    # Train
    result = train_model(
        train_data=train_data,
        eval_data=eval_data,
        config=config,
        device=device,
        progress=False,  # No progress bar in sweep
    )

    # Add task info to result
    result["config_name"] = task.config_name
    result["seed"] = task.seed
    result["epochs"] = task.epochs

    return result


def worker_fn(
    rank: int,
    world_size: int,
    tasks: list[SweepTask],
    results_queue: mp.Queue,
    use_amp: bool,
    n_gpus: int = 1,
):
    """Worker function for multiprocessing.

    Args:
        rank: Worker rank.
        world_size: Total number of workers.
        tasks: List of all tasks.
        results_queue: Queue to put results.
        use_amp: Whether to use AMP (not used with bf16).
        n_gpus: Number of available GPUs (for multi-GPU distribution).
    """
    # Distribute workers across GPUs: worker i uses GPU (i % n_gpus)
    gpu_id = rank % n_gpus
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    # Load data once per worker (on assigned GPU)
    train_data, eval_data = load_vix_dataset(
        window_size=250,
        n_eval=100,
        device=device,
    )

    # Process tasks assigned to this worker
    for i, task in enumerate(tasks):
        if i % world_size != rank:
            continue

        try:
            result = run_single_task(task, device, train_data, eval_data)
            results_queue.put(("success", task, result))
        except Exception as e:
            results_queue.put(("error", task, str(e)))


def run_sweep_sequential(
    tasks: list[SweepTask],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run sweep sequentially (single worker fallback).

    Args:
        tasks: List of tasks.
        device: Device to use.

    Returns:
        List of results.
    """
    # Load data once
    train_data, eval_data = load_vix_dataset(
        window_size=250,
        n_eval=100,
        device=device,
    )

    results = []
    for i, task in enumerate(tasks):
        print(f"Running task {i + 1}/{len(tasks)}: {task.config_name} seed={task.seed}")
        try:
            result = run_single_task(task, device, train_data, eval_data)
            results.append(result)
        except Exception as e:
            print(f"  Error: {e}")

    return results


def run_sweep_parallel(
    tasks: list[SweepTask],
    workers: int,
    use_amp: bool = False,
) -> list[dict[str, Any]]:
    """Run sweep with parallel workers using multiprocessing.

    Automatically distributes workers across all available GPUs.
    E.g., with 16 workers and 8 GPUs, each GPU runs 2 workers.

    Args:
        tasks: List of tasks.
        workers: Number of parallel workers.
        use_amp: Whether to use AMP.

    Returns:
        List of results.
    """
    # Set multiprocessing start method
    with suppress(RuntimeError):
        mp.set_start_method("spawn", force=True)

    # Detect available GPUs
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"GPUs available: {n_gpus}")
    if n_gpus > 1:
        print(f"Multi-GPU mode: {workers} workers across {n_gpus} GPUs")

    results_queue = mp.Queue()

    # Start workers
    processes = []
    for rank in range(workers):
        p = mp.Process(
            target=worker_fn,
            args=(rank, workers, tasks, results_queue, use_amp, n_gpus),
        )
        p.start()
        processes.append(p)

    # Collect results
    results = []
    completed = 0
    total = len(tasks)

    while completed < total:
        status, task, data = results_queue.get()
        completed += 1

        if status == "success":
            results.append(data)
            print(
                f"[{completed}/{total}] {task.config_name} s{task.seed}: "
                f"VaR={data['var_err_median']:.4f} ES={data['es_err_median']:.4f}"
            )
        else:
            print(f"[{completed}/{total}] {task.config_name} s{task.seed}: ERROR - {data}")

    # Wait for workers to finish
    for p in processes:
        p.join()

    return results


def parse_seeds(seeds_str: str) -> list[int]:
    """Parse seeds string to list of ints.

    Examples:
        "0,1,2,3" -> [0, 1, 2, 3]
        "0-4" -> [0, 1, 2, 3, 4]
        "0-4,10,15-17" -> [0, 1, 2, 3, 4, 10, 15, 16, 17]
    """
    seeds = []
    for part in seeds_str.split(","):
        if "-" in part:
            start, end = part.split("-")
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main():
    parser = argparse.ArgumentParser(description="Run FE-GAN sweep")

    # Config options (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config",
        type=Path,
        help="Path to single config YAML",
    )
    config_group.add_argument(
        "--sweep",
        type=Path,
        help="Path to sweep YAML (multiple configs)",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0-99",
        help="Seeds to run (e.g., '0,1,2' or '0-99')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Use automatic mixed precision",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="Enable cudnn.benchmark",
    )

    args = parser.parse_args()

    # Determine output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or Path(f"results/sweep_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []

    if args.config:
        # Single config sweep
        seeds = parse_seeds(args.seeds)
        config_name = args.config.stem

        # Load config to get default epochs
        cfg = OmegaConf.load(args.config)
        default_epochs = cfg.get("epochs", 100)
        epochs = args.epochs or default_epochs

        for seed in seeds:
            task = SweepTask(
                config_name=config_name,
                config_path=args.config,
                seed=seed,
                epochs=epochs,
                out_dir=out_dir / "per_run" / f"{config_name}_s{seed}",
                use_amp=args.use_amp,
                cudnn_benchmark=args.cudnn_benchmark,
            )
            tasks.append(task)

    else:
        # Multi-config sweep from YAML
        sweep_cfg = OmegaConf.load(args.sweep)

        for config_spec in sweep_cfg.configs:
            config_name = config_spec.name
            config_path = Path(config_spec.path)
            epochs = args.epochs or config_spec.get("epochs", 100)
            n_seeds = config_spec.get("seeds", 100)

            for seed in range(n_seeds):
                task = SweepTask(
                    config_name=config_name,
                    config_path=config_path,
                    seed=seed,
                    epochs=epochs,
                    out_dir=out_dir / "per_run" / f"{config_name}_s{seed}",
                    use_amp=sweep_cfg.get("use_amp", args.use_amp),
                    cudnn_benchmark=sweep_cfg.get("cudnn_benchmark", args.cudnn_benchmark),
                )
                tasks.append(task)

    print(f"Sweep: {len(tasks)} tasks")
    print(f"Output: {out_dir}")
    print(f"Workers: {args.workers}")
    print()

    start_time = time.time()

    # Run sweep
    if args.workers > 1:
        results = run_sweep_parallel(tasks, args.workers, args.use_amp)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = run_sweep_sequential(tasks, device)

    elapsed = time.time() - start_time

    # Save index CSV
    index_path = out_dir / "index.csv"
    with open(index_path, "w", newline="") as f:
        fieldnames = [
            "config_name",
            "seed",
            "epochs",
            "var_err_median",
            "var_err_mean",
            "es_err_median",
            "es_err_mean",
            "total_wall_seconds",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Save sweep metadata
    meta = {
        "n_tasks": len(tasks),
        "n_completed": len(results),
        "total_wall_seconds": elapsed,
        "workers": args.workers,
        **get_system_info(),
    }
    save_json(meta, out_dir / "sweep_meta.json")

    print()
    print(f"Completed {len(results)}/{len(tasks)} tasks in {elapsed:.1f}s")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
