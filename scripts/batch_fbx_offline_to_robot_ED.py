import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from rich import print
from tqdm import tqdm
import os
import numpy as np
import pickle


def load_optitrack_fbx_motion_file(motion_file):
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
    return motion_data


def offset_to_ground(retargeter: GMR, motion_data):
    offset = np.inf
    for human_data in motion_data:
        human_data = retargeter.to_numpy(human_data)
        human_data = retargeter.scale_human_data(
            human_data,
            retargeter.human_root_name,
            retargeter.human_scale_table,
        )
        human_data = retargeter.offset_human_data(
            human_data,
            retargeter.pos_offsets1,
            retargeter.rot_offsets1,
        )

        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if pos[2] < offset:
                offset = pos[2]

    return offset


def find_motion_files(root):
    """递归找到所有pkl motion文件"""
    motion_files = []
    for path in pathlib.Path(root).rglob("*.pkl"):
        motion_files.append(path)
    return motion_files


def retarget_single_motion(motion_file, save_path, retargeter, motion_fps):

    print(f"[green]Processing[/green]: {motion_file}")

    data_frames = load_optitrack_fbx_motion_file(motion_file)

    height_offset = offset_to_ground(retargeter, data_frames)
    retargeter.set_ground_offset(height_offset)

    qpos_list = []

    for smplx_data in data_frames:

        qpos = retargeter.retarget(smplx_data)

        qpos_list.append(qpos)

    root_pos = np.array([qpos[:3] for qpos in qpos_list])

    # wxyz -> xyzw
    root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])

    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    motion_data = {
        "fps": motion_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }

    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f)

    print(f"[cyan]Saved[/cyan] -> {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--motion_dir",
        required=True,
        type=str,
        help="Root directory containing motion files",
    )

    parser.add_argument(
        "--save_root",
        required=True,
        type=str,
        help="Root directory to save retargeted motions",
    )

    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "booster_t1",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--reset-retargeter-per-motion",
        action="store_true",
        help="每个动作使用新的 retargeter，使结果与单文件 fbx_offline_to_robot.py 一致",
    )

    args = parser.parse_args()

    motion_root = pathlib.Path(args.motion_dir)
    save_root = pathlib.Path(args.save_root)

    print(f"[yellow]Input motion root[/yellow]: {motion_root}")
    print(f"[yellow]Output motion root[/yellow]: {save_root}")

    # 找到所有motion文件
    motion_files = find_motion_files(motion_root)

    print(f"[green]Found {len(motion_files)} motion files[/green]")

    if len(motion_files) == 0:
        print("[red]No motion files found![/red]")
        exit()

    motion_fps = 120

    # 每个动作共用一个 retargeter 时只初始化一次；否则每段新建（与单文件脚本结果一致）
    if not args.reset_retargeter_per_motion:
        retargeter = GMR(
            src_human="fbx_ED",
            tgt_robot=args.robot,
            actual_human_height=1.6,
        )
    else:
        retargeter = None

    pbar = tqdm(motion_files, desc="Retargeting motions")

    for motion_file in pbar:

        rel_path = motion_file.relative_to(motion_root)
        save_path = save_root / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if args.reset_retargeter_per_motion:
            retargeter = GMR(
                src_human="fbx",
                tgt_robot=args.robot,
                actual_human_height=1.6,
            )

        retarget_single_motion(
            motion_file,
            save_path,
            retargeter,
            motion_fps,
        )

    print("[bold green]All motions processed successfully![/bold green]")