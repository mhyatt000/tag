import argparse

import genesis as gs
from tagEnv import TagEnv
import torch


def main(args):
    env = TagEnv(args)
    env.build()
    # TODO: Implement Actions
    for t in range(0, 600):
        # robot_1 will uncrouch repeatedly
        # robot_2 will crouch repeatedly
        if t % 60 == 0:
            if t != 0 and t % 120 != 0:
                env.robot_1.control_dofs_position(
                    position=torch.tile(
                        torch.tensor([0.1, 0.1, -0.1, -0.1, 0.6, 0.8, 0.6, 0.8, -1, -1, -1, -1], device=gs.device),
                        (args.n_envs, 1),
                    ),
                    dofs_idx_local=env.cfg.robotCfg.asset.local_dofs,
                    envs_idx=torch.tensor([i for i in range(0, args.n_envs)], device=gs.gpu),
                )
                env.robot_2.control_dofs_position(
                    position=torch.tile(
                        torch.tensor([0.1, 0.1, -0.1, -0.1, 1, 1.2, 1, 1.2, -1.7, -1.7, -1.7, -1.7], device=gs.device),
                        (args.n_envs, 1),
                    ),
                    dofs_idx_local=env.cfg.robotCfg.asset.local_dofs,
                    envs_idx=torch.tensor([i for i in range(0, args.n_envs)], device=gs.gpu),
                )
            else:
                env.robot_1.control_dofs_position(
                    position=torch.tile(
                        torch.tensor(
                            [0.1, 0.1, -0.1, -0.1, 0.8, 1.0, 0.8, 1.0, -1.5, -1.5, -1.5, -1.5], device=gs.device
                        ),
                        (args.n_envs, 1),
                    ),
                    dofs_idx_local=env.cfg.robotCfg.asset.local_dofs,
                    envs_idx=torch.tensor([i for i in range(0, args.n_envs)], device=gs.gpu),
                )
                env.robot_2.control_dofs_position(
                    position=torch.tile(
                        torch.tensor(
                            [0.1, 0.1, -0.1, -0.1, 0.8, 1.0, 0.8, 1.0, -1.5, -1.5, -1.5, -1.5], device=gs.device
                        ),
                        (args.n_envs, 1),
                    ),
                    dofs_idx_local=env.cfg.robotCfg.asset.local_dofs,
                    envs_idx=torch.tensor([i for i in range(0, args.n_envs)], device=gs.gpu),
                )
        env.robot_1.control_dofs_force(  # Send instruction to only the first environment's robot_1
            force=torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device=gs.device),
            dofs_idx_local=env.cfg.robotCfg.asset.local_dofs,
            envs_idx=torch.tensor([0], device=gs.gpu),
        )
        # TODO: Implement Progress Bar?
        env.step()

    env.record_data()


def get_args():
    """
    Some basic arguments as an example
    Many are excluded
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--logging_level", type=str, default="warning")
    parser.add_argument("-B", "--n_envs", type=int, default=1)
    parser.add_argument("--n_rendered", type=int, default=1)
    return parser.parse_args()


def check_args(args):
    logging_levels = ["debug", "info", "warning"]
    maximum_envs = 20000
    maximum_rendered_envs = 32

    # Check logging_level
    if args.logging_level not in logging_levels:
        sys.exit("ARGUMENT ERROR: LOGGING_LEVEL: Invalid logging_level parsed.")

    # Check n_envs
    if args.n_envs == 0:
        args.n_envs = 1
    if args.n_envs > maximum_envs:
        sys.exit("ARGUMENT ERROR: N_ENVS: Exceeds maximum environments.")
    if args.n_rendered > maximum_rendered_envs:
        sys.exit("ARGUMENT ERROR: N_RENDERED: Exceeds maximum rendered environments.")
    if args.n_rendered > args.n_envs:
        sys.exit("ARGUMENT ERROR: N_RENDERED: Rendered environments exceeds number of environments.")
    if args.headless is False:
        sys.exit("ARGUMENT ERROR: HEADLESS: Please run headless for now")

    return True


if __name__ == "__main__":
    args = get_args()
    check_args(args)
    main(args)
