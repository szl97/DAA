import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim_metric


def load_videos(video_dir):
    """
    读取 video_dir 下所有 .mp4 文件（按文件名排序）
    返回 [(frames_list, fps), ...]，frames_list 中每帧为 BGR ndarray
    """
    clips = []
    files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    for fname in files:
        path = os.path.join(video_dir, fname)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if frames:
            clips.append((frames, fps))
            print(f"加载: {fname}  帧数: {len(frames)}  FPS: {fps:.1f}")
    return clips


def save_video(frames, output_path, fps=30.0):
    """将帧列表写成 mp4"""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def generate_test_clips(video_dir, n=3, frames_per_clip=90, fps=30.0):
    """
    在 video_dir 生成 n 段彩色渐变合成测试片段，每段 frames_per_clip 帧
    用于在没有真实视频时验证算法
    """
    colors = [
        ((200, 80, 60), (60, 200, 80)),    # 蓝→绿
        ((60, 200, 80), (80, 60, 200)),    # 绿→红
        ((80, 60, 200), (200, 160, 60)),   # 红→青
    ]
    h, w = 480, 640
    for i in range(n):
        c1 = np.array(colors[i % len(colors)][0], dtype=np.float32)
        c2 = np.array(colors[i % len(colors)][1], dtype=np.float32)
        path = os.path.join(video_dir, f"test_clip_{i+1:02d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for j in range(frames_per_clip):
            t = j / (frames_per_clip - 1)
            color = ((1 - t) * c1 + t * c2).astype(np.uint8)
            frame = np.full((h, w, 3), color, dtype=np.uint8)
            # 加一个移动的白色圆，使光流有明显运动目标
            cx = int(w * (0.2 + 0.6 * (j / frames_per_clip)))
            cy = h // 2
            cv2.circle(frame, (cx, cy), 40, (255, 255, 255), -1)
            writer.write(frame)
        writer.release()
        print(f"生成测试片段: {path}")


def compute_frame_diff(f1, f2):
    """两帧像素平均绝对差（越小过渡越平滑）"""
    return float(np.mean(np.abs(f1.astype(np.float32) - f2.astype(np.float32))))


def compute_ssim(f1, f2):
    """两帧 SSIM（越高连贯性越好），转灰度后计算"""
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    return float(ssim_metric(g1, g2))
