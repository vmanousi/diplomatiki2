import argparse
import copy
from pathlib import Path

import optuna

from scripts.train import load_config, run_training


def build_trial_config(base_config, trial):
    """
    Copy the base config and override just the hyperparameters this
    trial is tuning. Everything else (model, backbone, augmentation,
    epochs, early stopping, etc.) stays fixed, matching the base config.
    """

    config = copy.deepcopy(base_config)
    training_cfg = config["training"]

    uses_discriminative_lr = (
        "backbone_learning_rate" in base_config["training"]
        and "head_learning_rate" in base_config["training"]
    )

    if uses_discriminative_lr:
        training_cfg["backbone_learning_rate"] = trial.suggest_float(
            "backbone_learning_rate", 1e-6, 1e-4, log=True
        )

        training_cfg["head_learning_rate"] = trial.suggest_float(
            "head_learning_rate", 1e-4, 1e-2, log=True
        )
    else:
        training_cfg["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True
        )

    training_cfg["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-4, 1e-1, log=True
    )

    config["experiment_name"] = (
        f"{base_config['experiment_name']}_optuna_trial{trial.number:03d}"
    )

    return config


def objective(trial, base_config, experiments_root):
    config = build_trial_config(base_config, trial)
    experiment_dir = experiments_root / f"trial{trial.number:03d}"

    def on_epoch_end(epoch, val_f1):
        # Lets Optuna kill a clearly unpromising trial early instead of
        # waiting for its own early_stopping_patience to trigger.
        trial.report(val_f1, step=epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    result = run_training(
        config=config,
        experiment_dir=experiment_dir,
        on_epoch_end=on_epoch_end,
    )

    return result["best_val_f1"]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Optuna hyperparameter search for a supervised GastroHUN "
            "config. Tunes learning rate(s) and weight_decay; every "
            "other setting in the base config (model, backbone, "
            "augmentation, epochs, early stopping, etc.) stays fixed. "
            "The objective is the best validation macro F1 reached "
            "during each trial's training run."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Base config to tune.",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help=(
            "Optuna study name. Defaults to "
            "'<base experiment_name>_optuna'. Reusing the same name "
            "resumes an existing study (via its SQLite file) instead of "
            "starting over — safe to resubmit if a job runs out of time."
        ),
    )

    args = parser.parse_args()

    base_config = load_config(args.config)

    # Groups every trial under outputs/experiments/optuna/<config file
    # name>/trialNNN/ — keeps tuning runs out of the main experiments
    # listing, and separate from any other config tuned this way later.
    config_name = Path(args.config).stem
    experiments_root = (
        Path("outputs") / "experiments" / "optuna" / config_name
    )

    study_name = args.study_name or f"{config_name}_optuna"

    storage_dir = Path("outputs") / "optuna"
    storage_dir.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{storage_dir / (study_name + '.db')}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )

    study.optimize(
        lambda trial: objective(trial, base_config, experiments_root),
        n_trials=args.n_trials,
        # One trial hitting an unexpected error shouldn't kill an
        # otherwise multi-hour study — it's marked FAILED and the study
        # moves on to the next trial.
        catch=(Exception,),
    )

    print("=" * 60)
    print("Optuna study:", study_name)
    print("Total trials so far:", len(study.trials))

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        print("No trials completed successfully — nothing to report.")
        return

    print("Best trial:")
    print("  value (best val macro F1):", study.best_trial.value)
    print("  params:", study.best_trial.params)
    print(
        "  output folder:",
        experiments_root / f"trial{study.best_trial.number:03d}",
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
