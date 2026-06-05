"""Unified command-line entry point.

Examples
--------
Train (on-node, default)::

    marl-patrol --mode on_node --num-episodes 1000

Train (every-timestep)::

    marl-patrol --mode every_timestep --num-episodes 1000

Evaluate a saved model and render GIFs::

    marl-patrol --mode on_node --test --render --suffix v2
"""

from marl_patrol.train import common


def main(argv=None):
    args = common.arg_parse(argv)
    common.setup_logging()
    common.set_seed(args.seed)
    cfgs = common.cfg_parse(args)
    writer = common.prepare_run(args, cfgs)

    if args.mode == "on_node":
        from marl_patrol.train import on_node as runner
    else:
        from marl_patrol.train import every_timestep as runner

    try:
        runner.main(args, cfgs, writer)
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
