#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""造一段 demo 素材：纯 ffmpeg lavfi 合成，不需要任何外部文件。

产物（都在本脚本旁边的 demo/ 里）：
    demo/demo.mp4   约 40 秒，4 段不同颜色的纯色画面 + 不同文字，模拟 4 张幻灯片
    demo/demo.srt   与视频时间轴对齐的 8 条字幕（通用讲座口吻）

跑法：
    python make_demo.py

需要 ffmpeg（带 drawtext/libfreetype）。找不到中文字体时自动退化成纯色拼接，
保证「能跑起来」优先。
"""
import os
import subprocess
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(ROOT, "demo")

# ffmpeg 路径只在 local_source.find_tool 里查一次，全仓库共用（不再各写一份）
from local_source import FFMPEG  # noqa: E402

W, H, FPS = 1280, 720, 25
SEG = 10                      # 每段秒数，4 段 → 40 秒

# 4 张「幻灯片」：底色 / 主标题 / 副标题
SLIDES = [
    ("0x1e3a5f", "课程导览", "这一节我们讲三个要点"),
    ("0x8a3b2f", "要点一　基本概念", "先把定义和适用范围说清楚"),
    ("0x2f5d3a", "要点二　核心方法", "关键在步骤顺序，不能颠倒"),
    ("0x4a3d6b", "要点三　典型例题", "跟着例题走一遍就明白了"),
]

# 8 条字幕：(起, 止, 文本)，与 40 秒画面严格对齐
CUES = [
    (0.0, 5.5, "各位同学好，这一节我们讲三个要点。"),
    (5.5, 11.0, "先看整体结构，再一条一条展开。"),
    (11.0, 16.5, "第一个要点是基本概念，重点是定义和适用范围。"),
    (16.5, 22.0, "第二个要点是核心方法，注意它的步骤顺序。"),
    (22.0, 27.5, "顺序不能乱，顺序错了结果就变了。"),
    (27.5, 33.0, "第三个要点，我们跟着一道典型例题走一遍。"),
    (33.0, 38.5, "课后请自己再推一遍，把容易错的地方标出来。"),
    (38.5, 40.0, "这一节先讲到这里。"),
]

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def find_font():
    for p in FONT_CANDIDATES:
        if os.path.isfile(p):
            return p
    return ""


def _filter_font_path(path):
    """给 drawtext 用的字体路径。

    Windows 盘符里的冒号在 filtergraph 里要转义，很麻烦；如果字体就在系统盘上，
    直接写成盘符相对路径（/Windows/Fonts/xxx.ttc）绕开这个坑。
    """
    drive, rest = os.path.splitdrive(path)
    if drive and drive.upper() == (os.environ.get("SystemDrive") or "C:").upper():
        return rest.replace("\\", "/")
    return path.replace("\\", "/").replace(":", "\\:")


def _drawtext(font, slide, idx):
    """一段画面的文字：主标题 + 副标题 + 右下角页码。"""
    if not font:
        return []
    _, title, sub = slide
    ff = _filter_font_path(font)
    footer = f"演示素材　第 {idx + 1} 页"
    return [
        (f"drawtext=fontfile={ff}:text='{title}':fontcolor=white:fontsize=64:"
         f"x=(w-text_w)/2:y=(h-text_h)/2-40"),
        (f"drawtext=fontfile={ff}:text='{sub}':fontcolor=0xd8e2ec:fontsize=34:"
         f"x=(w-text_w)/2:y=(h-text_h)/2+50"),
        (f"drawtext=fontfile={ff}:text='{footer}':fontcolor=0xaab6c2:fontsize=24:"
         f"x=w-text_w-48:y=h-text_h-40"),
    ]


def build_video(out_path, font=""):
    """4 路 lavfi 纯色源 → 各画自己的文字 → concat 成 40 秒 mp4。"""
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"]
    for color, _, _ in SLIDES:
        cmd += ["-f", "lavfi", "-i", f"color=c={color}:s={W}x{H}:r={FPS}:d={SEG}"]

    chains, labels = [], []
    for i, slide in enumerate(SLIDES):
        # 注意：输入标签 [i:v] 后面要直接跟滤镜名，中间不能有逗号（否则是个空滤镜）
        filters = _drawtext(font, slide, i)
        filters.append(f"trim=duration={SEG},setpts=PTS-STARTPTS[v{i}]")
        chains.append(f"[{i}:v]" + ",".join(filters))
        labels.append(f"[v{i}]")
    chains.append("".join(labels) + f"concat=n={len(SLIDES)}:v=1:a=0[out]")

    cmd += ["-filter_complex", ";".join(chains), "-map", "[out]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    return subprocess.run(cmd, capture_output=True)


def srt_text():
    def ts(sec):
        ms = int(round(sec * 1000))
        return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"

    blocks = []
    for i, (a, b, text) in enumerate(CUES, 1):
        blocks.append(f"{i}\n{ts(a)} --> {ts(b)}\n讲师：{text}\n")
    return "\n".join(blocks)


def main():
    if not (FFMPEG and os.path.isfile(FFMPEG)):
        print("!! 找不到 ffmpeg。装好后加进 PATH，或用 SHIKE_FFMPEG 指定路径。")
        return 1
    os.makedirs(DEMO_DIR, exist_ok=True)
    mp4 = os.path.join(DEMO_DIR, "demo.mp4")
    srt = os.path.join(DEMO_DIR, "demo.srt")

    font = find_font()
    print("ffmpeg:", FFMPEG)
    print("字体:", font or "（没找到中文字体，降级为纯色拼接）")

    r = build_video(mp4, font)
    if r.returncode != 0 or not os.path.isfile(mp4):
        err = (r.stderr or b"").decode("utf-8", "replace").strip()[:400]
        print("  带文字的合成失败，降级为纯色拼接：", err)
        r = build_video(mp4, "")
        if r.returncode != 0 or not os.path.isfile(mp4):
            print("!! ffmpeg 合成失败：",
                  (r.stderr or b"").decode("utf-8", "replace").strip()[:600])
            return 1
        print("  （已降级：画面只有纯色，没有文字）")

    with open(srt, "w", encoding="utf-8") as f:
        f.write(srt_text())

    size = os.path.getsize(mp4) / 1024
    print(f"完成：{mp4}  ({size:.0f} KB, {SEG * len(SLIDES)} 秒, {len(SLIDES)} 张幻灯片)")
    print(f"完成：{srt}  ({len(CUES)} 条字幕)")
    print("\n试跑：")
    print(f'  python lecture_notes.py demo{os.sep}demo.mp4 -t demo{os.sep}demo.srt '
          f'--title 演示讲座 -o 输出测试')
    return 0


if __name__ == "__main__":
    sys.exit(main())
