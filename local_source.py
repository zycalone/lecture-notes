#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地输入契约：本地视频 + 本地逐字稿 → 一个统一的 dict。

本模块只读本地文件，不发任何网络请求。视频时长用 ffprobe 从文件里读。

对外就两个函数：

    load_local(video_path, transcript_path=None) -> dict
        {
          "video":    视频绝对路径,
          "duration": 秒(float，读不到时为 0.0),
          "lines":    [{"t": 秒(float), "speaker": 发言人, "text": 文本}, ...],
          "title":    标题（默认取视频文件名）,
        }

    format_lines(lines) -> ["[HH:MM:SS] 发言人：文本", ...]

逐字稿支持三种格式：
    .srt  标准字幕块（序号 / 时间轴 / 文本）
    .vtt  WebVTT（跳过 WEBVTT 头与 NOTE 等段）
    .txt  每行 "[HH:MM:SS] 发言人：内容" 或 "HH:MM:SS 内容" 或纯文本行
          纯文本行没有时间戳时，按顺序在整段视频时长里均匀分配时间。
"""
import html
import os
import re
import shutil
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))


def find_tool(name, env_key):
    """ffmpeg / ffprobe 的唯一查找入口（全仓库只此一处）：环境变量 → PATH → 包内 tools/ffmpeg/bin。"""
    return (os.environ.get(env_key) or shutil.which(name)
            or os.path.join(ROOT, "tools", "ffmpeg", "bin", name + ".exe"))


FFMPEG = find_tool("ffmpeg", "SHIKE_FFMPEG")
FFPROBE = find_tool("ffprobe", "SHIKE_FFPROBE")

# 时间戳：HH:MM:SS(.mmm) 或 MM:SS(.mmm)，逗号/点都当小数分隔符（srt 用逗号）
_TS = re.compile(r"(?:(\d{1,3}):)?(\d{1,3}):(\d{1,2})(?:[.,](\d{1,3}))?")

# 只剥「已知的字幕行内标签」，不要把正文里的比较符号当标签吃掉：
#   <i> <b> <u> <s> <em> <strong> / 对应闭合标签 / <font color=…> /
#   <c.colorE5E5E5> / <v 讲师> / <00:00:01.000>
# 反例（必须原样保留）：字幕里写「x<0 且 y>0」「当 a<b 且 b>c 时」。
_TAG = re.compile(
    r"</?(?:i|b|u|s|em|strong|font|v|c)\s*>"
    r"|<font[^>]*>"
    r"|<c\.[^>]*>"
    r"|<v[^>]*>"
    r"|<\d{1,2}:\d{2}[^>]*>",
    re.I)

# 发言人前缀的判定：冒号前不超过 12 个字，且不能带句读，避免把「第一点：…」当成发言人
_SPEAKER = re.compile(r"^([^：:，。、！？；,.;!?]{1,12})\s*[：:]\s*(.+)$")


# ---------- 小工具 ----------
def _read_text(path):
    """按 UTF-8(BOM) → GB18030 顺序读，读不出来再兜底 replace。"""
    for enc in ("utf-8-sig", "gb18030"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _norm(text):
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")


def _sec(groups):
    """正则分组 → 秒(float)。"""
    h, m, s, frac = groups
    sec = (int(h) if h else 0) * 3600 + int(m) * 60 + int(s)
    if frac:
        sec += int(frac) / float(10 ** len(frac))
    return float(sec)


def _split_speaker(text):
    """"发言人：内容" → (发言人, 内容)；没有前缀就返回 ("", 整行)。

    先按白名单剥字幕标签，再 html.unescape 还原 &lt; &gt; &amp; 之类的实体。
    """
    text = html.unescape(_TAG.sub("", text)).strip()
    m = _SPEAKER.match(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text


def probe_duration(video):
    """ffprobe 取时长（秒）。取不到返回 0.0，不抛异常。"""
    exe = FFPROBE if os.path.isfile(FFPROBE) else (shutil.which("ffprobe") or "")
    if not exe:
        return 0.0
    try:
        r = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video],
            capture_output=True, timeout=120)
        out = (r.stdout or b"").decode("utf-8", "replace").strip().splitlines()
        return round(float(out[0]), 2) if out else 0.0
    except Exception:
        return 0.0


# ---------- 三种格式的解析 ----------
def _rows_to_lines(rows, duration):
    """[t, speaker, text] 列表 → 正式 lines；缺时间戳的行补时间。"""
    rows = [r for r in rows if (r[2] or "").strip()]
    if not rows:
        return []
    timed = any(r[0] is not None for r in rows)
    if not timed:
        # 全篇没有时间戳：按顺序在视频时长里均匀分配
        n = len(rows)
        step = (duration / n) if (duration and n) else 8.0
        for i, r in enumerate(rows):
            r[0] = round(i * step, 2)
    else:
        last = 0.0
        for r in rows:
            if r[0] is None:
                r[0] = round(last + 1.0, 2)
            last = r[0]
    return [{"t": float(r[0]), "speaker": (r[1] or "发言人").strip(), "text": r[2].strip()}
            for r in rows]


def _parse_srt(text, duration):
    rows = []
    for block in re.split(r"\n{2,}", _norm(text)):
        ls = [l for l in block.split("\n") if l.strip()]
        if not ls:
            continue
        if re.fullmatch(r"\d+", ls[0].strip()) and len(ls) > 1:
            ls = ls[1:]                      # 丢掉块的序号
        mt = _TS.search(ls[0])
        if not mt or "-->" not in ls[0]:
            continue
        body = " ".join(ls[1:]).strip()
        if not body:
            continue
        sp, tx = _split_speaker(body)
        rows.append([_sec(mt.groups()), sp, tx])
    return _rows_to_lines(rows, duration)


def _parse_vtt(text, duration):
    rows = []
    for block in re.split(r"\n{2,}", _norm(text)):
        ls = [l for l in block.split("\n") if l.strip()]
        if not ls:
            continue
        head = ls[0].strip().upper()
        if head.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue                          # 跳过文件头与注释段
        idx = next((i for i, l in enumerate(ls) if "-->" in l), None)
        if idx is None:
            continue                          # 没有时间轴的块（cue id 之类）
        mt = _TS.search(ls[idx])
        if not mt:
            continue
        body = " ".join(ls[idx + 1:]).strip()
        if not body:
            continue
        sp, tx = _split_speaker(body)
        rows.append([_sec(mt.groups()), sp, tx])
    return _rows_to_lines(rows, duration)


def _parse_txt(text, duration):
    rows = []
    for raw in _norm(text).split("\n"):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        t = None
        # 行首允许一个左括号，之后紧跟时间戳
        start = 1 if s[:1] in "[(" else 0
        mt = _TS.match(s, start)
        if mt:
            t = _sec(mt.groups())
            s = s[mt.end():].lstrip("])） \t-—:：")   # 剥掉 "] ） - 空格" 之类的粘连字符
        sp, tx = _split_speaker(s)
        rows.append([t, sp, tx])
    return _rows_to_lines(rows, duration)


def parse_transcript(path, duration=0.0):
    """按扩展名（实在认不出就看内容）解析逐字稿，返回 lines 列表。"""
    text = _read_text(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".srt":
        return _parse_srt(text, duration)
    if ext == ".vtt":
        return _parse_vtt(text, duration)
    if ext in (".txt", ".md", ".log", ""):
        return _parse_txt(text, duration)
    head = text[:400].upper()
    if "WEBVTT" in head:
        return _parse_vtt(text, duration)
    if "-->" in text:
        return _parse_srt(text, duration)
    return _parse_txt(text, duration)


# ---------- 对外接口 ----------
def load_local(video_path, transcript_path=None):
    """本地视频（+ 可选逐字稿）→ {"video", "duration", "lines", "title"}。"""
    video = os.path.abspath(video_path)
    if not os.path.isfile(video):
        raise FileNotFoundError(f"找不到视频文件：{video}")
    duration = probe_duration(video)

    lines = []
    if transcript_path:
        tp = os.path.abspath(transcript_path)
        if not os.path.isfile(tp):
            raise FileNotFoundError(f"找不到逐字稿文件：{tp}")
        lines = parse_transcript(tp, duration)

    title = os.path.splitext(os.path.basename(video))[0].strip() or "讲座"
    return {"video": video, "duration": duration, "lines": lines, "title": title}


def format_lines(lines):
    """lines → ["[HH:MM:SS] 发言人：文本", ...]"""
    def hms(sec):
        s = int(float(sec))
        return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

    return [f"[{hms(it['t'])}] {it.get('speaker') or '发言人'}：{it['text']}" for it in lines]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: local_source.py <video> [transcript]")
        sys.exit(1)
    d = load_local(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print("视频:", d["video"])
    print("时长: %.2f 秒" % d["duration"])
    print("标题:", d["title"])
    print("逐字稿:", len(d["lines"]), "句")
    for line in format_lines(d["lines"])[:10]:
        print(" ", line)
