import os
from utils import load_videos, save_video, generate_test_clips
from synthesize import optical_flow_blend

VIDEO_DIR = "videos"
OUTPUT_DIR = "output"
TRANSITION_FRAMES = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 若 videos/ 为空，自动生成合成测试片段
if not any(f.endswith(".mp4") for f in os.listdir(VIDEO_DIR)):
    print("videos/ 目录为空，自动生成合成测试片段...")
    generate_test_clips(VIDEO_DIR, n=3)

clips = load_videos(VIDEO_DIR)
print(f"\n共加载 {len(clips)} 段视频，开始光流融合...\n")

frames = optical_flow_blend(clips, transition_frames=TRANSITION_FRAMES)

output_path = os.path.join(OUTPUT_DIR, "synthesized.mp4")
save_video(frames, output_path, fps=clips[0][1])
print(f"输出: {output_path}")
