import cv2
import numpy as np
from utils import compute_frame_diff, compute_ssim

FLOW_PARAMS = dict(
    pyr_scale=0.5,
    levels=5,
    winsize=21,
    iterations=5,
    poly_n=7,
    poly_sigma=1.5,
    flags=0,
)

SEARCH_FRAMES = 60   # 在每段视频首尾各搜索60帧寻找最佳切点


def _smoothstep(t):
    """缓入缓出曲线：t²(3-2t)"""
    return t * t * (3.0 - 2.0 * t)


def _warp_frame(frame, flow, t):
    h, w = frame.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (gx + flow[..., 0] * t).astype(np.float32)
    map_y = (gy + flow[..., 1] * t).astype(np.float32)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _find_best_cut(frames_a, frames_b, search_n=SEARCH_FRAMES, scale=8):
    """
    最佳切点检测：
    在 frames_a 末尾 search_n 帧 × frames_b 开头 search_n 帧中，
    找出像素差异最小的帧对，作为融合切点。
    使用降采样加速对比（scale 倍）。
    返回 (idx_a, idx_b, min_diff)，均为原始数组下标。
    """
    end_a   = frames_a[-min(search_n, len(frames_a)):]
    start_b = frames_b[:min(search_n, len(frames_b))]
    h, w    = end_a[0].shape[:2]
    sw, sh  = max(1, w // scale), max(1, h // scale)

    small_a = [cv2.resize(f, (sw, sh)).astype(np.float32) for f in end_a]
    small_b = [cv2.resize(f, (sw, sh)).astype(np.float32) for f in start_b]

    best_diff = float('inf')
    best_i, best_j = len(end_a) - 1, 0

    for i, fa in enumerate(small_a):
        for j, fb in enumerate(small_b):
            d = np.mean(np.abs(fa - fb))
            if d < best_diff:
                best_diff, best_i, best_j = d, i, j

    idx_a = len(frames_a) - len(end_a) + best_i
    return idx_a, best_j, best_diff


def _lum_gain(src, target):
    """计算逐通道亮度增益，使 src 均值向 target 均值靠拢。"""
    gain = np.ones(3, dtype=np.float32)
    for c in range(3):
        s = float(src[..., c].mean())
        if s > 1.0:
            gain[c] = float(target[..., c].mean()) / s
    return gain


def _flow_blend(f_a, f_b, transition_frames):
    """
    双向 Farneback 光流 + 前向-后向一致性掩码 + 亮度过渡归一化。
    可信区域：双向 warp（各走一半，减少拉伸）再 alpha 混合；
    不可信区域：退回纯 alpha 混合，避免鬼影。
    亮度归一化：让颜色/亮度跳变随过渡帧均匀分布，消除突变感。
    """
    gray_a = cv2.cvtColor(f_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(f_b, cv2.COLOR_BGR2GRAY)

    flow_ab = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, **FLOW_PARAMS)
    flow_ba = cv2.calcOpticalFlowFarneback(gray_b, gray_a, None, **FLOW_PARAMS)

    # 前向-后向一致性掩码
    h, w = gray_a.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x = np.clip(gx + flow_ab[..., 0], 0, w - 1)
    map_y = np.clip(gy + flow_ab[..., 1], 0, h - 1)
    flow_ba_w = cv2.remap(flow_ba, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    err  = np.sqrt(np.sum((flow_ab + flow_ba_w) ** 2, axis=2))
    mask = (err < 3.0).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)[..., np.newaxis]  # (H,W,1)

    fa_f = f_a.astype(np.float32)
    fb_f = f_b.astype(np.float32)

    # 亮度增益：将 f_b 均值拉向 f_a，t=0 时全量修正，t=1 时无修正
    gain = _lum_gain(f_b, f_a)   # 使 f_b → f_a 亮度所需的增益

    frames = []
    for j in range(1, transition_frames):
        t = _smoothstep(j / transition_frames)

        # 双向 warp：f_a 向 B 走 t 步，f_b 向 A 走 (1-t) 步
        warp_a = _warp_frame(f_a, flow_ab,  t      ).astype(np.float32)
        warp_b = _warp_frame(f_b, flow_ba,  1.0 - t).astype(np.float32)

        # 亮度归一化：f_b 部分按 (1-t) 强度向 f_a 亮度靠拢
        lum_correction = 1.0 + (gain - 1.0) * (1.0 - t)   # t→1 时 correction→1
        warp_b_adj = np.clip(warp_b * lum_correction, 0, 255)
        fb_adj     = np.clip(fb_f   * lum_correction, 0, 255)

        # 可信区域：双向 warp 混合；不可信区域：纯 alpha 混合
        flow_blended = warp_a * (1.0 - t) + warp_b_adj * t
        pure_blended = fa_f   * (1.0 - t) + fb_adj     * t
        blended = flow_blended * mask + pure_blended * (1.0 - mask)
        frames.append(np.clip(blended, 0, 255).astype(np.uint8))
    return frames


def optical_flow_blend(clips, transition_frames=8):
    """
    智能切点检测 + Farneback光流 + Warp + Alpha 混合
    输入：clips = [(frames_list, fps), ...]
    输出：all_frames（最终合成帧序列）

    算法：
    1. 在相邻片段首尾各 SEARCH_FRAMES 帧中搜索视觉差异最小的帧对（智能切点）
    2. 在最优帧对上计算 Farneback 稠密光流
    3. 用光流 Warp f_a 生成运动补偿中间帧，再与 f_b Alpha 混合（共 transition_frames-1 帧）
    4. 若最佳切点帧差 < 10，直接硬切
    """
    if len(clips) < 2:
        print("片段数量不足，直接返回单段视频帧。")
        return clips[0][0] if clips else []

    # 统一分辨率为第一段视频的尺寸
    h0, w0 = clips[0][0][0].shape[:2]
    normalized = []
    for frames, fps in clips:
        if frames[0].shape[:2] != (h0, w0):
            frames = [cv2.resize(f, (w0, h0)) for f in frames]
        normalized.append((frames, fps))
    clips = normalized

    frames_list = [c[0] for c in clips]
    fps = clips[0][1]
    n  = len(clips)

    # 预计算所有过渡的最佳切点
    cut_points = []
    for i in range(n - 1):
        idx_a, idx_b, min_diff = _find_best_cut(frames_list[i], frames_list[i + 1])
        cut_points.append((idx_a, idx_b, min_diff))
        print(f"[过渡 {i+1}→{i+2}] 最佳切点帧差: {min_diff:.1f} "
              f"| 切自第{i+1}段第{idx_a}帧 / 第{i+2}段第{idx_b}帧")

    # 按切点组装最终帧序列
    all_frames = []
    total_diff = 0.0
    total_ssim = 0.0

    for i in range(n):
        # 本段的起止帧（由前后切点决定）
        start = cut_points[i - 1][1] if i > 0 else 0
        end   = cut_points[i][0]     if i < n - 1 else len(frames_list[i]) - 1

        all_frames.extend(frames_list[i][start:end + 1])

        if i < n - 1:
            f_a = frames_list[i][end]
            f_b = frames_list[i + 1][cut_points[i][1]]

            diff = compute_frame_diff(f_a, f_b)
            sim  = compute_ssim(f_a, f_b)
            total_diff += diff
            total_ssim += sim

            if diff < 10 or transition_frames <= 1:
                # 差异极小，直接硬切
                print(f"[过渡 {i+1}→{i+2}] 帧差极小({diff:.1f})，直接硬切")
            else:
                all_frames.extend(_flow_blend(f_a, f_b, transition_frames))

            print(f"[过渡 {i+1}→{i+2}] 切点帧差: {diff:.1f} | SSIM: {sim:.4f}")
            print("-" * 50)

    print(f"\n合成完成，共 {len(all_frames)} 帧  FPS: {fps:.1f}")
    if n > 1:
        print(f"平均切点帧差: {total_diff/(n-1):.1f} | "
              f"平均切点SSIM: {total_ssim/(n-1):.4f}")
    return all_frames
