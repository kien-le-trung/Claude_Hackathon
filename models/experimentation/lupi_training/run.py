from __future__ import annotations

import argparse

from lupi.learning import distill_student, evaluate, train_teacher
from lupi.pipeline import Settings, inspect, preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn LUPI knowledge distillation on EC3D")
    parser.add_argument("stage", choices=["inspect", "preprocess", "train-teacher", "distill",
                                         "evaluate", "run-all"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = Settings.load(args.config)
    if args.stage == "inspect":
        inspect(config)
    elif args.stage == "preprocess":
        preprocess(config)
    elif args.stage == "train-teacher":
        train_teacher(config)
    elif args.stage == "distill":
        distill_student(config)
    elif args.stage == "evaluate":
        evaluate(config)
    else:
        preprocess(config)
        train_teacher(config)
        distill_student(config)
        evaluate(config)


if __name__ == "__main__":
    main()
