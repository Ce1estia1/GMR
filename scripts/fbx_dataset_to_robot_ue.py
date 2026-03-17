import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from rich import print
from tqdm import tqdm
import os
import numpy as np
import pickle
from pathlib import Path

def load_optitrack_fbx_motion_file(motion_file):
 
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)

    # y-up 到 z-up 的变换四元数（绕 x 轴 -90°）
    transform_quat = np.array([np.sqrt(2)/2, -np.sqrt(2)/2, 0, 0])  # wxyz

    # 遍历所有帧
    for frame in motion_data:
        # 遍历字典中的所有键
        for key in frame:
            pos, quat = frame[key]

            # 转换位置: (x, y, z) → (x, z, -y)
            x, y, z = pos
            new_pos = np.array([x, z, -y])

            # 转换四元数: transform_quat × quat
            w1, x1, y1, z1 = transform_quat
            w2, x2, y2, z2 = quat
            new_quat = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])

            frame[key] = (new_pos, new_quat)

    return motion_data

def offset_to_ground(retargeter: GMR, motion_data):
    offset = np.inf
    for human_data in motion_data:
        human_data = retargeter.to_numpy(human_data)
        human_data = retargeter.scale_human_data(human_data, retargeter.human_root_name, retargeter.human_scale_table)
        human_data = retargeter.offset_human_data(human_data, retargeter.pos_offsets1, retargeter.rot_offsets1)
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            if pos[2] < offset:
                offset = pos[2]

    return offset

def find_all_pkl_files(root_dir):
    """
    递归查找所有子文件夹中的pkl文件
    返回相对路径列表
    """
    root_path = Path(root_dir)
    pkl_files = []

    for pkl_file in root_path.rglob("*.pkl"):
        # 获取相对于根目录的路径
        rel_path = pkl_file.relative_to(root_path)
        pkl_files.append(rel_path)

    return sorted(pkl_files)

def process_single_motion(retargeter, motion_data, motion_fps, tgt_robot,video_output_path):
    """
    处理单个动作文件，返回机器人动作数据
    如果指定了video_output_path，则录制视频
    """
    from general_motion_retargeting import RobotMotionViewer

    # 如果需要录制视频
    if video_output_path is not None:
        save_dir = os.path.dirname(video_output_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        robot_motion_viewer = RobotMotionViewer(
            robot_type=tgt_robot,
            motion_fps=motion_fps,
            transparent_robot=1,
            record_video=True,
            video_path=video_output_path,
            camera_follow=False,
        )

    qpos_list = []

    for i, smplx_data in enumerate(motion_data):
        # retarget
        qpos = retargeter.retarget(smplx_data)

        #可视化/录制
        if video_output_path is not None:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                rate_limit=False,
            )

        qpos_list.append(qpos)

    # 关闭视频录制器
    if video_output_path is not None:
        robot_motion_viewer.close()

    # 转换为numpy数组
    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])  # wxyz to xyzw
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    return {
        "fps": motion_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
    }

if __name__ == "__main__":

    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description="批量处理OptiTrack FBX动作文件")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="包含pkl动作文件的根目录（支持递归查找子文件夹）",
    )

    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01"],
        default="unitree_g1",
    )

    parser.add_argument(
        "--record_video",
        action="store_true",
        default=True,
        help="是否录制视频"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录（保留原始文件夹层级结构）",
    )

    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.8,
        help="人体高度（米）"
    )

    parser.add_argument(
        "--motion_fps",
        type=int,
        default=120,
        help="动作帧率"
    )
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 查找所有pkl文件
    print(f"扫描目录: {args.input_dir}")
    pkl_files = find_all_pkl_files(args.input_dir)
    print(f"找到 {len(pkl_files)} 个pkl文件")

    # 初始化重定向器（只初始化一次）
    retargeter = GMR(
        src_human="fbx_offline_ue",
        tgt_robot=args.robot,
        actual_human_height=args.actual_human_height,
    )

    # 批量处理
    success_count = 0
    fail_count = 0
    failed_files = []

    for rel_pkl_path in tqdm(pkl_files, desc="处理动作文件"):
        try:
            # 拼接完整路径
            input_pkl_path = input_dir / rel_pkl_path

            # 构建输出路径（保留层级结构）
            output_pkl_path = output_dir / rel_pkl_path
            output_pkl_path = output_pkl_path.with_suffix(".pkl")  # 确保扩展名正确

            # 构建视频输出路径
            video_output_path = None
            if args.record_video:
                video_output_path = output_pkl_path.with_suffix(".mp4")

            # 加载动作数据
            data_frames = load_optitrack_fbx_motion_file(input_pkl_path)

            # 计算地面偏移
            height_offset = offset_to_ground(retargeter, data_frames)
            retargeter.set_ground_offset(height_offset)

            # 处理动作
            motion_data = process_single_motion(
                retargeter,
                data_frames,
                args.motion_fps,
                args.robot,
                video_output_path
            )

            # 保存机器人动作
            os.makedirs(output_pkl_path.parent, exist_ok=True)
            with open(output_pkl_path, "wb") as f:
                pickle.dump(motion_data, f)

            success_count += 1

        except Exception as e:
            fail_count += 1
            failed_files.append(str(rel_pkl_path))
            print(f"\n处理失败: {rel_pkl_path}, 错误: {str(e)}")
            continue

    # 打印统计信息
    print("\n" + "="*50)
    print("批处理完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    if fail_count > 0:
        print("\n失败的文件:")
        for f in failed_files:
            print(f"  - {f}")
    print("="*50)
