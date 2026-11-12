#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地录播 → 课件截图 + 图文笔记 + docx/pdf + 单文件回看页。

只吃本地文件：一段本地视频 + 一份本地逐字稿（字幕可以用可选 ASR 现场生成）。
不联网抓取、不依赖任何平台的登录态，全程离线（只有可选的笔记环节会调用你自己
配置的 OpenAI 兼容接口）。

用法：
  python lecture_notes.py <video.mp4> [-t <transcript.srt|vtt|txt>] [--title 名称] [-o 输出目录]
                                   [--asr-model small] [--lang zh]

不给 -t 时自动尝试可选 ASR（faster-whisper）；两者都没有就打印一行指引、退出码 2
（课件截图、回看页、元信息照常产出）。

退出码（多个坑同时命中取优先级高的：1 > 3 > 2 > 0）：
  0  一切正常
  1  输入不可用（视频不存在 / 读不出时长、不是可读的视频、找不到 ffmpeg）
  2  缺逐字稿（没给 -t 且 ASR 不可用）——课件截图、回看页、元信息照常产出
  3  产物不完整（0 张幻灯片；或配了模型 key 但笔记/docx 生成失败）
不给 -o 时输出到 ./输出/<视频名>/

流程：
  1. load_local() 读本地视频 + 逐字稿 → {"video","duration","lines","title"}
     （没给字幕就调 asr.transcribe()，见可选依赖）
  2. ffmpeg 按固定间隔抽帧，numpy+PIL 按画面差异去重 → 幻灯片/
     （装了 OCR 就再按文字相似度二次去重，并把文字写进 index.json）
  3. 逐字稿 + 幻灯片时间点交给大模型 → 笔记.md → 笔记_图文版.md
  4. md2docx 导出 docx（找得到 LibreOffice 才出 pdf）
  5. 生成 回看.html：单文件、离线、双击就能跟着视频看逐字稿

可选依赖（都不装也能跑通，只是少一些能力）：
  faster-whisper        没有逐字稿时自动转写；没装 → 明确提示 + 退出码 2
  rapidocr-onnxruntime  课件截图 OCR，让音频/板书文字可搜；没装 → 静默跳过
"""
import argparse
import difflib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote

import numpy as np
from PIL import Image

# ffmpeg/ffprobe 的查找只此一处（local_source.find_tool），这里不再重复定义
from local_source import FFMPEG, format_lines, load_local

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------- 配置 ----------
# 所有路径都以本文件所在目录为基准（ROOT），换机器不用改代码：
#   SHIKE_FFMPEG / SHIKE_FFPROBE  ffmpeg / ffprobe 路径（默认先找 PATH，再找 tools/ffmpeg/bin）
#   SHIKE_PROXY                   OpenAI 兼容接口地址    SHIKE_MODEL  模型名
#   SHIKE_PROXY_KEY               接口密钥（其次读同目录 _local_key.txt）
ROOT = os.path.dirname(os.path.abspath(__file__))

PROXY = os.environ.get("SHIKE_PROXY") or "https://api.openai.com/v1"
MODEL = os.environ.get("SHIKE_MODEL") or "gpt-4o-mini"

# 退出码：优先级 1 > 3 > 2 > 0（同时命中取更大的坑）
RC_OK = 0              # 一切正常
RC_INPUT = 1           # 输入不可用：视频不存在 / 不是可读的视频 / 找不到 ffmpeg
RC_NO_TRANSCRIPT = 2   # 缺逐字稿：没给 -t 且 ASR 不可用（其它产物照常）
RC_INCOMPLETE = 3      # 产物不完整：0 张幻灯片，或配了 key 但笔记/docx 失败
_RC_PRIORITY = {RC_OK: 0, RC_NO_TRANSCRIPT: 1, RC_INCOMPLETE: 2, RC_INPUT: 3}


def _worse(a, b):
    """两个退出码取优先级高的那个（1 > 3 > 2 > 0）。"""
    return a if _RC_PRIORITY.get(a, 0) >= _RC_PRIORITY.get(b, 0) else b


SLIDE_STEP = 15          # 抽帧间隔（秒，≥180s 的长片），可用 SHIKE_SLIDE_STEP 覆盖
SLIDE_STEP_MID = 8       # 60~180s 的中等片长用 8 秒
SLIDE_STEP_SHORT = 3     # <60s 的短片用 3 秒（15 秒一帧在 3 秒视频上抽不出任何帧）
SLIDE_DIFF = float(os.environ.get("SHIKE_SLIDE_DIFF") or 3.2)  # 画面变化阈值上限，越大越不敏感
SLIDE_MAX = 110          # 最多保留张数，可用 SHIKE_SLIDE_MAX 覆盖
MIN_GAP = 45             # 相邻关键帧最小间隔（秒），短片上会自动收紧

# --- 可选 OCR 相关的调参（仅在装了 rapidocr 时生效，没装完全不影响） ---
OCR_TEXT_SIM = 0.92      # 相邻两帧文字相似度 > 这个值 → 判为同一页被切了两次
DIFF_FLOOR = 0.15         # 自适应阈值下限：实测同一页相邻帧差为 0.000，留一点余量给压缩噪声
DIFF_RATIO = 0.15         # 自适应阈值 = 本段视频最大帧差 × 该比例（不超过 SLIDE_DIFF）
OCR_CANDIDATE_DIFF = 0.05 # 装了 OCR 时的候选阈值：只滤掉真正一模一样（0.000）的帧
OCR_MAX_FRAMES = int(os.environ.get("SHIKE_OCR_MAX") or 120)  # 送去文字比对的帧数上限
OCR_MERGE_DIFF = 10.0    # 再做二次去重时的「画面差也小」上限（同一套 64x36 灰度均差）
OCR_MIN_CHARS = 6        # 文字太短（< 这么多字）不参与相似度判断，避免「1」「2」被误合并
OCR_MAX_CHARS = 400      # 每页写进 index.json / 提示词的文字上限，防止整页 PPT 灌爆 json
WEB_COPY_LIMIT = 300 * 1024 * 1024   # 视频不在输出目录时，≤300MB 复制一份让目录自包含


def _load_key():
    """密钥不入库：环境变量 SHIKE_PROXY_KEY 优先，其次同目录 _local_key.txt。"""
    for env in ("SHIKE_PROXY_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(env)
        if v and v.strip():
            return v.strip()
    p = os.path.join(ROOT, "_local_key.txt")
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8-sig") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""


PROXY_KEY = _load_key()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hhmmss(sec):
    s = int(float(sec))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def safe_name(s, fallback="讲座"):
    s = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", (s or "").strip())
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s[:60] or fallback


def _tool(path, name):
    """确认可执行文件在，不在就返回空串。"""
    return path if (path and os.path.isfile(path)) else (shutil.which(name) or "")


# ---------- 1. 抽帧 + 去重（保留原算法：间隔抽帧 + 画面差异阈值） ----------
def slide_params(duration):
    """(抽帧间隔, 最小间隔)。

    step 按片长分档：≥180s → 15 秒；60~180s → 8 秒；<60s → 3 秒。
      原因：fps=1/15 在 3.2 秒的视频上一帧都抽不出来（0 张幻灯片），短片必须加密。
    gap  = min(MIN_GAP, max(2, duration / 4))：短片按比例收紧但最少留 2 秒，
      否则 8 秒的视频只剩开头一张、中途换页全部丢掉。
    环境变量 SHIKE_SLIDE_STEP 一旦显式设置，就直接用它当 step（不再分档）。
    """
    try:
        step = int(os.environ.get("SHIKE_SLIDE_STEP") or 0)
    except ValueError:
        step = 0
    dur = float(duration or 0.0)
    if step <= 0:
        if dur and dur < 60:
            step = SLIDE_STEP_SHORT
        elif dur and dur < 180:
            step = SLIDE_STEP_MID
        else:
            step = SLIDE_STEP
    step = max(1, step)
    # gap：怕「同一页反复轻微闪动」被记成好几张，但也不能粗到把正常换页吞掉。
    # 实测 31 秒、4 页的课件，换页间隔 6~9 秒，原来 /4 算出的 8 秒直接把第 4 页丢了；
    # 改成 duration/12 并至少留一个 step，长视频仍由 MIN_GAP 封顶。
    gap = MIN_GAP if dur <= 0 else max(float(step), min(float(MIN_GAP), dur / 12.0))
    return step, gap


def slide_max():
    """最多保留张数（SHIKE_SLIDE_MAX 可覆盖，测试时用来压低阈值）。"""
    try:
        v = int(os.environ.get("SHIKE_SLIDE_MAX") or 0)
    except ValueError:
        v = 0
    return v if v > 0 else SLIDE_MAX


def thin_out(kept, limit=None):
    """按倍数抽样把关键帧压到 limit 张以内（ceil(len/limit) 步长）。

    原来是 kept[::len//SLIDE_MAX]：111 张时步长 1，等于没抽样。
    """
    limit = limit or slide_max()
    if limit <= 0 or len(kept) <= limit:
        return kept
    stride = max(1, -(-len(kept) // limit))       # ceil(len / limit)
    return kept[::stride]


def thumb(path):
    return np.asarray(Image.open(path).convert("L").resize((64, 36)), dtype=np.float32)


def _thumb_diff(p1, p2):
    """两张图的画面差异（同一套 64x36 灰度均差，和抽帧去重同一个量纲）。"""
    try:
        return float(np.mean(np.abs(thumb(p1) - thumb(p2))))
    except Exception:
        return 999.0


def _text_sim(a, b):
    """两段 OCR 文字的相似度 0~1。任一方太短就返回 0（不参与合并判断）。"""
    a, b = (a or "").strip(), (b or "").strip()
    if len(a) < OCR_MIN_CHARS or len(b) < OCR_MIN_CHARS:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _ocr_text(ocr_fn, path):
    """跑一次 OCR；任何异常都吞掉返回空串（OCR 是可选能力，不能影响主流程）。"""
    if not ocr_fn:
        return ""
    try:
        return (ocr_fn(path) or "").strip()[:OCR_MAX_CHARS]
    except Exception:
        return ""


_MKDTEMP_OK = [True]     # tempfile.mkdtemp 建的目录在本机能不能写（第一次失败就记下来）


def _frame_tmpdir(fallback_root):
    """抽帧中间文件放哪儿：优先系统临时目录下的短路径（%TEMP%/ln_xxx）。

    原因：输出目录一长（实测 248 字符），ffmpeg 写 f_0001.jpg 就报
    「Could not open file」→ 一帧都抽不出来。临时目录短得多，能绕开路径长度问题。
    先用 tempfile.mkdtemp(prefix="ln_")；它建的目录权限是 0700，在受限环境下
    可能连自己都写不进去，所以再退一步用普通 mkdir 建一个同样短的 ln_* 目录，
    两个都不行才回退到输出目录下的 _frames（老行为，路径长的时候会失败）。
    """
    stamp = f"ln_{os.getpid()}_{int(time.time() * 1000) % 100000}"
    cands = []
    if _MKDTEMP_OK[0]:
        try:
            cands.append(tempfile.mkdtemp(prefix="ln_"))
        except Exception as e:
            log(f"  （mkdtemp 建不出临时目录：{type(e).__name__} {e}）")
            _MKDTEMP_OK[0] = False
    cands.append(os.path.join(tempfile.gettempdir(), stamp))
    for d in cands:
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, "probe.tmp")
            with open(probe, "wb") as f:
                f.write(b"ok")
            os.remove(probe)
            return d
        except Exception as e:
            log(f"  （临时目录不可用，换下一个：{type(e).__name__} {e}）")
            _MKDTEMP_OK[0] = False       # 这个环境的 mkdtemp 目录写不进去，本进程不再试
            shutil.rmtree(d, ignore_errors=True)
    d = os.path.join(fallback_root, "_frames")
    os.makedirs(d, exist_ok=True)
    return d


def extract_slides(video, imgdir, duration=0.0, ocr_fn=None):
    """按固定间隔抽帧 → 画面差异去重 → （可选）OCR 文字二次去重 → 存 <imgdir>/*.jpg。

    返回 (items, 抽帧数)。items 每项：{"index","second","t","time","file","text"}；
    text 是这一页的 OCR 文字（没装 OCR 时为空串）。
    """
    os.makedirs(imgdir, exist_ok=True)
    step, gap = slide_params(duration)
    tmpdir = _frame_tmpdir(os.path.dirname(os.path.abspath(imgdir)))
    for f in os.listdir(tmpdir):
        if f.endswith(".jpg"):
            try:
                os.remove(os.path.join(tmpdir, f))
            except OSError:
                pass
    try:
        log(f"  抽帧（每 {step} 秒一帧）...")
        t0 = time.time()
        r = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", video,
                            "-vf", f"fps=1/{step},scale=1024:-1", "-q:v", "5",
                            os.path.join(tmpdir, "f_%04d.jpg")],
                           check=False, capture_output=True)
        files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".jpg"))
        if not files:
            err = (r.stderr or b"").decode("utf-8", "replace").strip()[:300]
            log(f"  !! ffmpeg 没抽出任何帧{('：' + err) if err else ''}")
            return [], 0
        # 先算一遍相邻帧的画面差，再据此定阈值。
        # 固定阈值 3.2 对「同一套模板、只换几行字」的课件太钝：实测换页时差异只有 0.6~1.0，
        # 会被整段合并成一张。所以改成跟着这段视频自己的尺度走（自适应），并留 SLIDE_DIFF 兜底。
        thumbs, diffs = [], []
        for i, fn in enumerate(files):
            th = thumb(os.path.join(tmpdir, fn))
            thumbs.append(th)
            diffs.append(999.0 if i == 0 else float(np.mean(np.abs(th - thumbs[i - 1]))))
        body = [d for d in diffs[1:] if d < 900]
        hi = max(body) if body else 0.0
        thr = min(SLIDE_DIFF, max(DIFF_FLOOR, DIFF_RATIO * hi))
        # 装了 OCR 时，把「画面几乎没变、但也不完全相同」的帧也留下来交给文字比对，
        # 否则像素层面根本分不出「换页」和「同一页」
        loose = min(thr, OCR_CANDIDATE_DIFF) if ocr_fn else thr
        if ocr_fn:
            # 有 OCR 文字比对时，就不该再靠「最小间隔」这种粗手段挡新页了：
            # 实测 8 秒间隔会把第 4 张课件直接吞掉（24s 那页距上一页只有 6 秒）
            gap = min(gap, max(float(step), 1.0))
        log(f"  抽到 {len(files)} 帧，用时 {time.time() - t0:.0f}s，开始去重"
            f"（阈值 {thr:.2f}{'，OCR 候选 ' + format(loose, '.2f') if ocr_fn else ''}，"
            f"最小间隔 {gap:.0f}s）")

        kept, prev_t = [], -10 ** 9
        for i, fn in enumerate(files):
            t = i * step
            diff = diffs[i]
            if diff >= loose and (t - prev_t) >= gap:
                kept.append((t, os.path.join(tmpdir, fn)))
                prev_t = t
        if len(kept) > slide_max():
            before = len(kept)
            kept = thin_out(kept)
            log(f"  关键帧超过上限（{before} > {slide_max()}），按倍数抽样"
                f" → 剩 {len(kept)} 张")
        if ocr_fn and len(kept) > OCR_MAX_FRAMES:
            before = len(kept)
            kept = thin_out(kept, OCR_MAX_FRAMES)
            log(f"  OCR 候选帧过多（{before}），抽稀到 {len(kept)} 帧再做文字比对")
        log(f"  关键帧 {len(kept)} 张")

        # OCR（可选）：给每一张保留帧取文字。装了才算，没装就是空表，流程一模一样。
        texts = {}
        if ocr_fn:
            log(f"  OCR 识别 {len(kept)} 张课件文字（可选依赖）...")
            t_ocr = time.time()
            for _t, _p in kept:
                texts[_p] = _ocr_text(ocr_fn, _p)
            got = sum(1 for v in texts.values() if v)
            log(f"  OCR 完成：{got}/{len(kept)} 张有文字，用时 {time.time() - t_ocr:.0f}s")

            # 文本相似度二次去重：文字几乎一样 + 画面差别也小 → 同页被切两次，保留后一张
            if len(kept) > 1:
                merged_keep, merged = [], 0
                for item in kept:
                    if merged_keep:
                        p_prev = merged_keep[-1][1]
                        sim = _text_sim(texts.get(p_prev, ""), texts.get(item[1], ""))
                        if sim > OCR_TEXT_SIM and _thumb_diff(p_prev, item[1]) < OCR_MERGE_DIFF:
                            merged_keep[-1] = item      # 保留后一张（内容通常更完整）
                            merged += 1
                            continue
                    merged_keep.append(item)
                if merged:
                    kept = merged_keep
                    log(f"  OCR 文本去重：合并 {merged} 张重复页（相似度 > {OCR_TEXT_SIM} 且画面差小）"
                        f" → 剩 {len(kept)} 张")

        items = []
        for idx, (t, path) in enumerate(kept, 1):
            h, m, s = int(t) // 3600, int(t) % 3600 // 60, int(t) % 60
            name = f"{idx:02d}_{h:02d}-{m:02d}-{s:02d}.jpg"
            Image.open(path).save(os.path.join(imgdir, name), quality=88)
            items.append({"index": idx, "second": t, "t": float(t),
                          "time": f"{h:02d}:{m:02d}:{s:02d}",
                          "file": name, "text": texts.get(path, "")})
        with open(os.path.join(imgdir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)
        return items, len(files)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def caption_for(items, lines):
    """给每张截图配一句「那一刻在讲什么」，用于图文版图注。

    复用回看页那套 _near_sentence（取该时刻、含略靠后 2 秒，正在讲的那句）。
    原实现取的是「t+30 秒窗口内最后一句」，于是 0 秒那张配上的是 27.5 秒才说的话。
    降级：一条词都用不上时，退回时间上最接近的那一句。
    """
    rows = [{"t": float(it.get("t", 0.0) or 0.0), "text": it.get("text") or ""}
            for it in lines or []]
    for it in items:
        sec = float(it.get("second", it.get("t", 0.0)) or 0.0)
        best = _near_sentence(rows, sec) if rows else ""
        if not best and rows:
            best = min(rows, key=lambda r: abs(r["t"] - sec)).get("text") or ""
        best = best.strip()
        it["caption"] = (best[:46] + "…") if len(best) > 46 else (best or "（无语音）")
    return items


# ---------- 2. 大模型笔记 ----------
def _fold_system(messages):
    """部分代理会丢弃 system 消息，统一折进第一条 user 消息。"""
    sys_txt = "\n\n".join(str(m.get("content") or "") for m in messages
                          if m.get("role") == "system").strip()
    rest = [dict(m) for m in messages if m.get("role") != "system"]
    if not sys_txt:
        return rest or [{"role": "user", "content": ""}]
    if rest:
        rest[0]["content"] = sys_txt + "\n\n" + str(rest[0].get("content") or "")
        return rest
    return [{"role": "user", "content": sys_txt}]


def _session():
    """共用一个会话，并且关掉 urllib3 自带的内部重试。

    默认的 urllib3 会在连接层自己重试 3 轮，再叠上我们这层重试，
    地址打错时要等一分多钟才报错——自己控制节奏更可预测。
    """
    import requests
    from requests.adapters import HTTPAdapter

    global _SESSION_OBJ
    try:
        return _SESSION_OBJ
    except NameError:
        pass
    sess = requests.Session()
    adapter = HTTPAdapter(max_retries=0)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    _SESSION_OBJ = sess
    return sess


def _chat_once(body, stream):
    import requests

    if not stream:
        r = _session().post(PROXY.rstrip("/") + "/chat/completions",
                            headers={"Authorization": f"Bearer {PROXY_KEY}"},
                            json=body, timeout=900)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    out = []
    with _session().post(PROXY.rstrip("/") + "/chat/completions",
                         headers={"Authorization": f"Bearer {PROXY_KEY}"},
                         json=body, stream=True, timeout=1800) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", "replace")
            if not line.startswith("data: "):
                continue
            d = line[6:].strip()
            if d == "[DONE]":
                break
            try:
                j = json.loads(d)
            except Exception:
                continue
            piece = ((j.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
            if piece:
                out.append(piece)
    return "".join(out).strip()


def _http_hint(code, err):
    """把接口返回的错误码翻成人话（用户看得懂、能自己动手改的那种）。"""
    hint = {
        400: "请求被拒（模型名写错、额度不足或内容被拦截，检查 SHIKE_MODEL）",
        401: "密钥无效（检查 SHIKE_PROXY_KEY 或同目录 _local_key.txt）",
        403: "密钥没权限访问这个模型",
        404: "接口地址不对（SHIKE_PROXY 一般要以 /v1 结尾，如 https://xxx/v1）",
        413: "内容太长（逐字稿过大，试试更小的模型上下文或先分段）",
        429: "被限流了（等一会儿再跑，或换个 key）",
    }.get(code, "接口报错")
    return f"{hint}；返回 {code}：{str(err)[:160]}"


def chat(messages, temperature=0.3, stream=False):
    """调用 OpenAI 兼容接口。

    重试策略分三档，避免「打错地址白等一分多钟」和「密钥错了还傻等 71 秒」：
      4xx（401/403/404/400…，429 除外）→ 立刻失败，报明确的错
      连不上/超时            → 2/5/10/20 秒，快试四轮
      5xx / 429             → 3/8/20/40 秒，正常退避
    """
    import requests

    messages = _fold_system(messages)
    body = {"model": MODEL, "temperature": temperature, "messages": messages, "stream": stream}
    err = None
    tries = 0
    for attempt in range(4):
        tries = attempt + 1
        try:
            return _chat_once(body, stream)
        except requests.HTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", 0) or 0
            if 400 <= code < 500 and code != 429:
                raise RuntimeError(_http_hint(code, e)) from e
            err = e
            wait = (3, 8, 20, 40)[min(attempt, 3)]
            log(f"    !! 接口返回 {code or '错误'}，{wait}s 后重试（第 {attempt + 1}/4 次）")
            time.sleep(wait)
        except Exception as e:
            err = e
            if attempt >= 1:      # 连不上只试两轮：地址打错不会自己变好，别让用户干等
                break
            log(f"    !! 连不上接口（{type(e).__name__}），2s 后重试")
            time.sleep(2)
    if isinstance(err, requests.ConnectionError):
        raise RuntimeError(
            f"连不上接口（试了 {tries} 次）：检查 SHIKE_PROXY 地址、网络或代理 —— "
            f"{str(err)[:120]}")
    raise RuntimeError(f"接口调用失败（试了 {tries} 次）：{type(err).__name__}: {str(err)[:160]}")
    raise err


def fix_latex(md):
    md = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n$$\n" + m.group(1).strip() + "\n$$\n", md, flags=re.S)
    md = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", md, flags=re.S)
    md = md.replace("\\[", "\n$$\n").replace("\\]", "\n$$\n")
    md = md.replace("\\(", "$").replace("\\)", "$")
    md = md.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return md


META_LINE = re.compile(
    r"^\s*(如果需要|如需|需要我|要不要|我可以|希望这份|以上就是|这份笔记|如有需要|"
    r"你还想|还想我|是否需要|可以继续|后续可以)")

CHUNK_SYSTEM = """你是课堂笔记提炼员。用户给你一段带时间戳的语音识别逐字稿（可能很长），
你要把它压成高密度知识点条目，供后续合并成分节笔记。

只输出 Markdown 片段（不要大标题、不要寒暄、不要结尾征求意见），按讲课顺序写：

- **概念/公式**：写清定义式、条件、结论；公式用 $...$（行内）或 $$...$$（独立行）的 LaTeX。
- **例题**：逐题记录——题目原话、思路、关键步骤算式、答案；每题前面标时间戳 `[hh:mm:ss]`。
- **讲师强调**：易错点、口诀、记忆技巧，用「⚠️」开头。
- **口误修正**：语音识别错的专业术语按学科常识改正，拿不准的加（疑似）。

看不到的内容不要编。宁可条目少，不要凑字数。"""

MERGE_SYSTEM = """你是资深课程笔记主编。用户给你一份「分段提炼稿」（同一堂课按时间分段整理的知识点条目），
请合并成一份结构完整、可直接复习和转发的笔记。

硬性要求：
1. 严格去重：同一知识点在不同分段重复出现，合并成一处，不要重复陈述。
2. 必须包含下列章节，标题文字照抄，顺序不变：
   # <标题> · 笔记
   > 元信息（讲师/时长/来源/逐字稿覆盖范围）
   ## 一、内容全景        （一段话脉络 + 「章节→核心问题→结论」表格）
   ## 二、知识点详解      （按讲课顺序分 ### 小节：定义 / 性质公式 / 易错点 ⚠️ / 口诀）
   ## 三、例题与解法      （逐题：题目 → 思路 → 关键步骤 → 答案，标时间点）
   ## 四、重点与复习清单  （必背结论 / 常考题型 / 自测问题 5~8 个，不给答案）
   ## 五、待确认 / 存疑处
3. 公式一律用 $...$ 或 $$...$$（KaTeX），禁止 \\( \\) 和 \\[ \\]。
4. 定义域值域、公式对照、等价无穷小这类内容一律做成表格。
5. 篇幅：短课 4000~6000 字；2 小时以上 9000~13000 字。宁详勿略。
6. 结尾严禁出现「如果需要我可以…」「需要我继续…」这类话；最后一行就是正文最后一条内容。"""


def _chunk(lines, max_chars=12000):
    out, cur, n = [], [], 0
    for l in lines:
        if cur and n + len(l) > max_chars:
            out.append(cur)
            cur, n = [], 0
        cur.append(l)
        n += len(l) + 1
    if cur:
        out.append(cur)
    return out


def _strip_meta(md):
    keep = [l for l in md.rstrip().splitlines() if not META_LINE.match(l)]
    while keep and (not keep[-1].strip() or keep[-1].strip() in ("---", "***")):
        keep.pop()
    return "\n".join(keep).strip()


def _slide_text_block(items):
    """把有 OCR 文字的课件页整理成「该页文字摘录」，给笔记提示词当参考。

    只有装了 OCR 且真的识别出字时才返回内容；否则返回空串（提示词保持原样）。
    """
    rows = []
    for it in items or []:
        txt = (it.get("text") or "").strip()
        if txt:
            rows.append(f"[{it.get('time', '')}] 该页文字摘录：{txt}")
    return "\n".join(rows)


def gen_note(lines, title, outdir, items=None):
    """逐字稿 → 笔记.md（长稿先分段提炼再合并）。items 有 OCR 文字时一并喂给模型。"""
    transcript = "\n".join(lines)
    slides = _slide_text_block(items)
    slides_part = ("\n\n课件页文字摘录（OCR，按时间顺序；用来判断每页在讲什么、校正专业术语）：\n"
                   + slides + "\n") if slides else ""
    if len(transcript) > 24000:
        parts = _chunk(lines)
        log(f"  逐字稿 {len(transcript)} 字，分 {len(parts)} 段精读")
        from concurrent.futures import ThreadPoolExecutor

        def one(idx_part):
            idx, ch = idx_part
            head, tail = ch[0][:11], ch[-1][:11]
            txt = chat([{"role": "system", "content": CHUNK_SYSTEM},
                        {"role": "user", "content": f"逐字稿第 {idx + 1} 段（{head} 至 {tail}）：\n\n"
                                                    + "\n".join(ch)}], temperature=0.2)
            log(f"    段 {idx + 1}/{len(parts)} 提炼完成 {len(txt)} 字")
            return f"\n\n## 分段 {idx + 1}（{head} 至 {tail}）\n\n{txt}"

        with ThreadPoolExecutor(max_workers=2) as ex:
            merged = "".join(ex.map(one, enumerate(parts)))
        md = chat([{"role": "system", "content": MERGE_SYSTEM + (
                       "\n\n若另给了「课件页文字摘录」，可用它校正术语、补全板书要点。" if slides else "")},
                   {"role": "user", "content": f"课程标题：{title}\n\n分段提炼稿：\n{merged}{slides_part}"}],
                  temperature=0.3, stream=True)
    else:
        md = chat([{"role": "system", "content": MERGE_SYSTEM + "\n\n下面是完整逐字稿，请直接整理：" + (
                       "\n若另给了「课件页文字摘录」，可用它校正术语、补全板书要点。" if slides else "")},
                   {"role": "user", "content": f"课程标题：{title}\n\n逐字稿：\n\n{transcript}{slides_part}"}],
                  temperature=0.3, stream=True)
    if slides:
        log(f"  提示词里附了 {slides.count('该页文字摘录')} 页课件文字摘录（OCR）")
    md = _strip_meta(fix_latex(md))
    path = os.path.join(outdir, "笔记.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    log(f"  笔记: {len(md)} 字")
    return path


def _looks_like_table(text):
    """模型偶尔不听话、把正文当时间轴返回；检查一下再决定要不要拼进笔记。"""
    rows = [ln for ln in (text or "").splitlines() if ln.strip().startswith("|")]
    return len(rows) >= 2


def gen_timeline(lines):
    prompt = ("下面是一份带时间戳的逐字稿。请只输出一张 Markdown 表格，两列：时间 | 这一段讲了什么"
              "（14 字以内）。每 5~8 分钟一行，按时间顺序覆盖全程。"
              "时间列必须抄逐字稿里的 hh:mm:ss 原样（例如 00:24:01），不要写成区间。"
              "不要任何前言后语，不要代码块包裹。\n\n" + "\n".join(lines))
    t = chat([{"role": "user", "content": prompt}], temperature=0.2)
    return re.sub(r"^```(?:markdown)?\s*|\s*```$", "", t).strip()


# ---------- 3. 讲解时间点 ↔ 课件帧 对齐 ----------
def build_illustrated(note_path, items, outdir, video_name=""):
    """把每张课件截图插到最合适的笔记章节里，生成 笔记_图文版.md。"""
    with open(note_path, encoding="utf-8") as f:
        md = f.read()
    heads = [h.strip() for h in re.findall(r"^##\s+(.*)$", md, re.M)]
    times = "\n".join(f"{it['file']} = {it['time']}" for it in items)
    prompt = f"""把每张课件截图分配到最合适的笔记章节。只输出 JSON：
{{"assign":[{{"file":"01_00-00-00.jpg","heading":"一、内容全景"}}]}}

heading 必须逐字取自下列二级标题；每张都要分配，不能遗漏。

标题：
{os.linesep.join(heads)}

截图（文件名 = 时刻）：
{times}
"""
    assign = []
    try:
        c = chat([{"role": "user", "content": prompt}], temperature=0.1)
        c = re.sub(r"^```(?:json)?\s*|\s*```$", "", c)
        assign = json.loads(re.search(r"\{.*\}", c, re.S).group(0)).get("assign", [])
    except Exception as e:
        log(f"  !! 章节分配失败（截图将全部放在文末）: {type(e).__name__} {e}")

    by_file = {it["file"]: it for it in items}
    bucket = {}
    for a in assign:
        if a.get("file") in by_file:
            bucket.setdefault((a.get("heading") or "").strip(), []).append(a["file"])
    assigned = {f for v in bucket.values() for f in v}
    rest = [it["file"] for it in items if it["file"] not in assigned]

    out, used = [], 0

    def emit(files):
        nonlocal used
        for f in sorted(files, key=lambda x: by_file[x]["second"]):
            it = by_file[f]
            out.append(f"![{it['time']} 课件](幻灯片/{f})")
            out.append(f"*{it['time']}　{it['caption']}*")
            out.append("")
            used += 1

    for line in md.splitlines():
        out.append(line)
        m = re.match(r"^##\s+(.*)$", line)
        if m and bucket.get(m.group(1).strip()):
            out.append("")
            emit(bucket[m.group(1).strip()])

    if rest:
        out += ["", "## 附：其余课件截图", ""]
        emit(rest)

    out.insert(1, f"\n> 图文版：正文中穿插 {len(items)} 张课件截图，图注是视频时间点"
                  + (f"（源视频 `{video_name}`）" if video_name else "") + "。\n")
    path = os.path.join(outdir, "笔记_图文版.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    log(f"  图文版: 插入 {used} 张截图")
    return path


# ---------- 4. 导出 docx / pdf ----------
def export_office(md_path, outdir, plain_md=None):
    """导出 docx（找得到 LibreOffice 才顺带出 pdf）。

    plain_md：纯文字版笔记。给了就一起导出成 笔记.docx，
    这样「只要 Word」的人不用去图多的图文版里挑。
    """
    try:
        import md2docx
    except Exception as e:
        log(f"  !! 导入 md2docx 失败，跳过 docx/pdf: {e}")
        return []
    jobs = []
    if plain_md and os.path.isfile(plain_md):
        jobs.append((plain_md, "笔记"))
    jobs.append((md_path, "笔记_图文版"))
    made = []
    for src_md, stem in jobs:
        try:
            made += md2docx.export(src_md, outdir, stem=stem, pdf=True)
        except Exception as e:
            log(f"  !! {stem}.docx 导出失败: {type(e).__name__} {e}")
    for name in made:
        p = os.path.join(outdir, name)
        if os.path.isfile(p):
            log(f"  {name}: {os.path.getsize(p) / 1048576:.1f} MB")
    return made


# ---------- 4.5 可选 ASR（没给 -t 时自动兜底） ----------
# 两档文案分开：① 根本没装 faster-whisper ② 装了但拿不到模型权重（首次需要联网下载）
ASR_HINT_NOT_INSTALLED = (
    "没有逐字稿：没有给 -t 字幕文件，也没装可选依赖 faster-whisper。"
    "两条路选一条：pip install faster-whisper（装完会自动转写，"
    "可用 --asr-model small 调模型），或者用 -t 字幕文件 指定已有字幕。")
ASR_HINT_NO_WEIGHTS = (
    "没有逐字稿：本机已装 faster-whisper，但拿不到模型权重"
    "（首次使用需要联网下载一次，或设置本地模型目录后离线用）。"
    "三条路选一条：① 联网后重跑，先用 --asr-model tiny 试（体积最小，约 75MB）；"
    "② --asr-model <本地模型目录> 指向已经下好的权重目录（离线可用）；"
    "③ 用 -t 字幕文件 指定已有字幕。"
    "本次课件截图、回看页、元信息照常产出。")
ASR_HINT_FAILED = (
    "没有逐字稿：ASR 跑起来了但失败（原因见上面日志）。"
    "可以换 --asr-model 再试，或者用 -t 字幕文件 指定已有字幕。"
    "本次课件截图、回看页、元信息照常产出。")


def asr_transcribe(video, model_size="small", lang="zh", device=None):
    """尝试可选 ASR。

    成功 → (lines, 字幕来源标签, "")；失败 → (None, "", 给用户看的提示文案)。
    """
    try:
        import asr
    except Exception as e:                     # 文件丢了/语法问题，都算不可用
        log(f"  可选 ASR 模块导入失败：{type(e).__name__} {e}")
        return None, "", ASR_HINT_NOT_INSTALLED
    if not asr.available():
        return None, "", ASR_HINT_NOT_INSTALLED
    try:
        lines = asr.transcribe(video, model_size=model_size, language=lang, on_log=log,
                               device=device)
    except asr.AsrWeightsUnavailable as e:
        log(f"  ASR 拿不到模型权重：{e}")
        return None, "", ASR_HINT_NO_WEIGHTS
    except asr.AsrUnavailable as e:
        log(f"  ASR 不可用：{e}")
        return None, "", ASR_HINT_NOT_INSTALLED
    except Exception as e:
        log(f"  !! ASR 失败（{type(e).__name__}: {str(e)[:120]}）")
        return None, "", ASR_HINT_FAILED
    if not lines:
        log("  !! ASR 没识别出任何文字（视频没有音轨？）")
        return None, "", ASR_HINT_FAILED
    return lines, asr.describe(model_size), ""


# ---------- 5. 回看页（单文件 HTML，无任何额外依赖，每次运行都生成） ----------
def _url_path(p):
    """路径 → 能直接放进 <video src> 的 URL 形态：各段 quote，保留 / : ~。

    不转义的话，视频名里的 # 会被浏览器当成 fragment、? 会被当成查询串，
    例如「第1讲#1 demo.mp4」会变成去请求 ./第1讲 —— 视频根本播不出来。
    """
    return quote(p.replace(os.sep, "/"), safe="/:~")


def _web_video_src(video, outdir):
    """<video src> 用什么：同目录/输出目录内 → 相对路径；否则复制一份或退回绝对 file://。

    规则（自己定，理由是让「拷走整个输出目录」还能播）：
      1. 视频就在输出目录里            → "./文件名"（最稳，整个目录可搬走）
      2. 视频在输出目录的子目录里      → 相对路径
      3. 视频在别处且 ≤ WEB_COPY_LIMIT → 复制一份进输出目录，仍用相对路径
      4. 视频在别处且很大/不允许复制   → 绝对 file:// 路径（唯一缺点：换机器或搬目录后失效）
    环境变量：SHIKE_WEB_COPY=0 关掉复制；SHIKE_WEB_COPY_LIMIT=字节数 改阈值。
    所有分支都过 _url_path：文件名里的 # ? % 空格都不会被当成 URL 语法。
    """
    vac, out = os.path.abspath(video), os.path.abspath(outdir)
    if os.path.dirname(vac) == out:
        return "./" + _url_path(os.path.basename(vac)), "同目录相对路径"

    rel = os.path.relpath(vac, out)
    if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
        return _url_path(rel), "输出目录内相对路径"

    try:
        limit = int(os.environ.get("SHIKE_WEB_COPY_LIMIT") or WEB_COPY_LIMIT)
    except ValueError:
        limit = WEB_COPY_LIMIT
    want_copy = (os.environ.get("SHIKE_WEB_COPY") or "1").strip().lower() not in (
        "0", "false", "no", "off")
    try:
        size = os.path.getsize(vac)
    except OSError:
        size = 0

    if want_copy and size and size <= limit:
        dst = os.path.join(out, os.path.basename(vac))
        if os.path.abspath(dst) != vac:
            try:
                shutil.copy2(vac, dst)
                log(f"  视频不在输出目录，已复制一份（{size / 1048576:.1f} MB）："
                    f"{os.path.basename(vac)}")
                return "./" + _url_path(os.path.basename(vac)), \
                    f"已复制进输出目录（{size / 1048576:.1f} MB）"
            except OSError as e:
                log(f"  !! 复制视频失败，改用绝对路径：{e}")

    uri = "file:///" + _url_path(vac)
    why = "视频较大，没复制" if want_copy else "SHIKE_WEB_COPY=0"
    return uri, f"绝对 file:// 路径（{why}；换机器或搬目录后会失效）"


REVIEW_CSS = """*{box-sizing:border-box}
:root{--bg:#12161c;--fg:#e9eef5;--dim:#8b98a8;--line:#243040;--acc:#5cb8ff;--card:#1a212b}
html,body{margin:0;background:var(--bg);color:var(--fg);
  font:16px/1.75 "Microsoft YaHei","微软雅黑","PingFang SC","Hiragino Sans GB",
  "Noto Sans CJK SC","Source Han Sans SC",system-ui,sans-serif}
header{padding:14px 18px 6px}
h1{margin:0 0 4px;font-size:19px;font-weight:600;line-height:1.4}
.meta{color:var(--dim);font-size:13px}
.wrap{display:flex;gap:16px;align-items:flex-start;padding:12px 18px 40px}
.left{flex:0 0 54%;position:sticky;top:12px}
video{display:block;width:100%;height:auto;background:#000;border-radius:10px}
.tip{color:var(--dim);font-size:12.5px;margin-top:8px;word-break:break-all}
.right{flex:1 1 auto;min-width:0;max-height:calc(100vh - 130px);overflow-y:auto;
  border-left:1px solid var(--line);padding-left:14px}
a.s{display:flex;gap:10px;text-decoration:none;color:var(--fg);padding:6px 9px;
  border-radius:8px;border-left:3px solid transparent}
a.s time{color:var(--dim);font-variant-numeric:tabular-nums;font-size:13px;flex:0 0 auto;padding-top:3px}
a.s span{font-size:15.5px;word-break:break-word}
a.s:hover{background:var(--card)}
a.s.on{background:#1d3550;border-left-color:var(--acc)}
a.s.on time{color:var(--acc)}
a.sl{display:block;text-decoration:none;color:var(--dim);margin:14px 0;padding:6px;
  border:1px solid var(--line);border-radius:10px;background:var(--card)}
a.sl img{display:block;width:100%;height:auto;border-radius:6px}
a.sl .cap{display:block;font-size:12.5px;margin-top:6px;color:var(--dim);word-break:break-word}
a.sl:hover{border-color:var(--acc)}
.empty{color:var(--dim);font-size:14px}
@media (max-width:820px){
  .wrap{flex-direction:column;padding:10px 12px 32px;gap:10px}
  .left{position:static;flex:none;width:100%}
  .right{max-height:none;overflow:visible;border-left:0;padding-left:0;width:100%}
}"""

REVIEW_JS = """(function(){
  var v=document.getElementById('v'), list=document.getElementById('list');
  if(!v||!list){return;}
  var nodes=list.querySelectorAll('a[data-t]');
  var ts=[], i;
  for(i=0;i<nodes.length;i++){ ts.push(parseFloat(nodes[i].getAttribute('data-t'))||0); }
  var cur=-1, lockUntil=0;
  function seek(t){
    if(isNaN(t)){return;}
    v.currentTime=t;
    lockUntil=0;
    try{ v.play(); }catch(e){}
  }
  list.addEventListener('scroll', function(){ lockUntil=Date.now()+2500; }, {passive:true});
  function paint(){
    var t=v.currentTime+0.25, idx=-1;
    for(var k=0;k<ts.length;k++){ if(ts[k]<=t){ idx=k; } else { break; } }
    if(idx===cur){ return; }
    if(cur>=0 && nodes[cur]){ nodes[cur].className=nodes[cur].className.replace(/\\s*\\bon\\b/,''); }
    cur=idx;
    if(idx<0){ return; }
    nodes[idx].className=nodes[idx].className+' on';
    if(Date.now()>lockUntil && nodes[idx].scrollIntoView){
      nodes[idx].scrollIntoView({block:'center', behavior:'smooth'});
    }
  }
  v.addEventListener('timeupdate', paint);
  v.addEventListener('seeked', paint);
  v.addEventListener('loadedmetadata', paint);
  list.addEventListener('click', function(e){
    var a=e.target;
    while(a && a!==list && !(a.getAttribute && a.getAttribute('data-t'))){ a=a.parentNode; }
    if(!a || a===list || !a.getAttribute){ return; }
    e.preventDefault();
    seek(parseFloat(a.getAttribute('data-t')));
  });
  paint();
})();"""


def _sentence_row(idx, it):
    """一句逐字稿 → 一个可点击的 <a>（data-t 是秒，点击就跳）。"""
    t = float(it.get("t", 0.0) or 0.0)
    spk = (it.get("speaker") or "").strip()
    text = it.get("text") or ""
    if spk and spk not in ("发言人",):
        text = f"{spk}：{text}"
    return (f'<a href="#" data-t="{t:.3f}" class="s">'
            f'<time>{hhmmss(t)}</time><span>{html.escape(text, quote=False)}</span></a>')


def _near_sentence(lines, t):
    """找「这一页出现的时刻」在讲的那句：取 t 附近（含略靠后 2 秒）最后一句。"""
    best, best_t = "", -10 ** 9
    for it in lines or []:
        s = float(it.get("t", 0.0) or 0.0)
        if s <= t + 2.0 and s >= best_t:
            best, best_t = (it.get("text") or ""), s
    if not best and lines:
        best = lines[0].get("text") or ""
    return best.strip()


def _slide_row(it, fallback=""):
    """一张课件图 → 章节分隔（点图也跳到那一刻）。优先显示这一页的 OCR 文字。"""
    t = float(it.get("second", it.get("t", 0.0)) or 0.0)
    fn = it.get("file") or ""
    cap = (it.get("text") or "").strip() or (fallback or "").strip()
    if len(cap) > 70:
        cap = cap[:70] + "…"
    cap_html = (f'<span class="cap"><b>{hhmmss(t)}</b>'
                f'{("　" + html.escape(cap, quote=False)) if cap else ""}</span>')
    return (f'<a href="#" data-t="{t:.3f}" class="sl">'
            f'<img src="幻灯片/{html.escape(fn, quote=True)}" alt="{hhmmss(t)} 课件" loading="lazy">'
            f'{cap_html}</a>')


def gen_review_html(outdir, video, lines_dicts, items, title, source="", ocr_on=False,
                    duration=0.0):
    """生成 <outdir>/回看.html：单文件、纯原生 JS、离线、双击即用。"""
    src, how = _web_video_src(video, outdir)

    rows = []
    for i, it in enumerate(lines_dicts or []):
        rows.append((float(it.get("t", 0.0) or 0.0), 1, "s", i))
    for j, it in enumerate(items or []):
        rows.append((float(it.get("second", it.get("t", 0.0)) or 0.0), 0, "sl", j))
    rows.sort(key=lambda r: (r[0], r[1]))            # 同一秒：先图后句

    body = []
    for _t, _k, kind, i in rows:
        if kind == "s":
            body.append(_sentence_row(i, lines_dicts[i]))
        else:
            body.append(_slide_row(items[i], _near_sentence(lines_dicts, _t)))
    if not body:
        body.append('<p class="empty">这次没生成到逐字稿和课件图。</p>')

    meta = [f"时长 {hhmmss(duration or _duration_of(lines_dicts, items))}",
            f"逐字稿 {len(lines_dicts or [])} 句",
            f"课件 {len(items or [])} 张"]
    if source:
        meta.append(f"字幕来源：{source}")
    meta.append("OCR：" + ("启用" if ocr_on else "未安装"))
    meta.append(f"生成于 {time.strftime('%Y-%m-%d %H:%M')}")

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title, quote=False)} · 回看</title>
<style>
{REVIEW_CSS}
</style>
</head>
<body>
<header>
<h1>{html.escape(title, quote=False)}</h1>
<div class="meta">{html.escape(" · ".join(meta), quote=False)}</div>
</header>
<main class="wrap">
<section class="left">
<video id="v" controls preload="metadata" playsinline src="{html.escape(src, quote=True)}"></video>
<div class="tip">点右边任意一句跳转；点课件图也跳转。视频地址：{html.escape(how, quote=False)}</div>
</section>
<section class="right" id="list">
{chr(10).join(body)}
</section>
</main>
<script>
{REVIEW_JS}
</script>
</body>
</html>
"""
    path = os.path.join(outdir, "回看.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    kb = os.path.getsize(path) / 1024
    log(f"  回看页: 回看.html（{kb:.0f} KB，{len(items or [])} 张课件，视频用{how}）")
    return path


def _duration_of(lines_dicts, items):
    """没记时长时给个下界估计：最后一句的时间。"""
    best = 0.0
    for it in (lines_dicts or []):
        best = max(best, float(it.get("t", 0.0) or 0.0))
    for it in (items or []):
        best = max(best, float(it.get("second", it.get("t", 0.0)) or 0.0))
    return best


# ---------- 6. 主流程 ----------
def process(video, transcript=None, title=None, outdir=None, asr_model="small", lang="zh",
            asr_device=None):
    """跑完整流程。返回 (输出目录, 退出码)；输入不可用时返回 None（main 转成退出码 1）。

    退出码规则见文件头：0 正常 / 1 输入不可用 / 2 缺逐字稿 / 3 产物不完整。
    """
    if not os.path.isfile(video):
        log("!! 找不到视频:", video)
        return None
    if not _tool(FFMPEG, "ffmpeg"):
        log("!! 找不到 ffmpeg，请装好并加进 PATH，或用 SHIKE_FFMPEG 指定路径")
        return None

    src = load_local(video, transcript)
    if not src.get("duration"):
        # ffprobe 读不到时长 = 不是能解的视频（也可能文件是坏的/空壳）
        log("!! 不是可读的视频（ffprobe 读不到时长）：", src["video"])
        return None
    video_name = os.path.basename(src["video"])
    title = safe_name(title or src["title"])
    outdir = os.path.abspath(outdir or os.path.join("输出", title))

    # 逐字稿来源：有就用；没有（含「给了 -t 但解析出 0 句」）就试可选 ASR；都失败 → 提示 + 退出码 2
    lines_dicts = src["lines"]
    rc = RC_OK
    if lines_dicts:
        source = "本地文件"
    else:
        if transcript:
            log(f"  字幕文件里没解析出任何句子（空文件或格式不认识）：{transcript}")
        log("  没有拿到逐字稿，尝试可选 ASR ...")
        got, src_label, hint = asr_transcribe(src["video"], model_size=asr_model, lang=lang,
                                          device=asr_device)
        if got:
            lines_dicts, source = got, src_label
        else:
            log("!!", hint)
            lines_dicts, source = [], "无"
            rc = RC_NO_TRANSCRIPT

    os.makedirs(outdir, exist_ok=True)
    lines = format_lines(lines_dicts)
    log(f"处理: {video_name}  时长 {hhmmss(src['duration'])}  逐字稿 {len(lines)} 句"
        + (f"  来源 {source}" if source else ""))
    log(f"输出目录: {outdir}（{len(outdir)} 字符）")
    if len(outdir) > 240:
        log(f"  !! 输出路径过长（{len(outdir)} > 240 字符）：部分系统上 ffmpeg / Word "
            f"会写不进去，建议换一个短一点的 -o 目录（例如放到 D:\\笔记\\课题名）")

    # 逐字稿（首行是注释，标清楚这份稿子哪儿来的）
    tpath = os.path.join(outdir, "逐字稿.txt")
    if lines:
        with open(tpath, "w", encoding="utf-8") as f:
            f.write(f"# 字幕来源：{source}\n")
            f.write("\n".join(lines) + "\n")
        log(f"  逐字稿: {len(lines)} 句 → 逐字稿.txt（字幕来源：{source}）")
    else:
        log("  （没有逐字稿，跳过 逐字稿.txt）")

    # 可选 OCR：装了才用，没装 ocr_fn 就是 None，抽帧逻辑完全不变
    ocr_fn, ocr_on = None, False
    try:
        import ocr as ocr_mod
        if ocr_mod.available():
            ocr_fn, ocr_on = ocr_mod.extract, True
        else:
            log(f"  OCR 未安装（课件文字不参与检索）：{ocr_mod.INSTALL_HINT}")
    except Exception as e:
        log(f"  OCR 模块不可用，跳过：{type(e).__name__} {e}")

    # 课件截图
    items, n_frames = [], 0
    try:
        items, n_frames = extract_slides(src["video"], os.path.join(outdir, "幻灯片"),
                                         src["duration"], ocr_fn=ocr_fn)
        caption_for(items, lines_dicts)
    except Exception as e:
        log(f"  !! 抽帧失败（继续）: {type(e).__name__} {e}")
    if not items:
        log("  !! 产物不完整：一张幻灯片都没抽到（退出码 3）")
        rc = _worse(rc, RC_INCOMPLETE)

    # 笔记（没有 key 就优雅跳过）
    illustrated = None
    if not PROXY_KEY:
        log("  跳过笔记生成：没有配置模型密钥（SHIKE_PROXY_KEY 或同目录 _local_key.txt）")
    elif not lines:
        log("  跳过笔记生成：没有逐字稿")
    else:
        try:
            note_path = gen_note(lines, title, outdir, items=items)
            md = open(note_path, encoding="utf-8").read()
            if "视频时间轴" not in md:
                tl = gen_timeline(lines)
                if _looks_like_table(tl):
                    md += "\n\n---\n\n## 视频时间轴\n\n" + tl + "\n"
                    with open(note_path, "w", encoding="utf-8") as f:
                        f.write(md)
                else:
                    log("  !! 时间轴没返回表格，已跳过（免得把正文再抄一遍）")
            illustrated = build_illustrated(note_path, items, outdir, video_name) if items else note_path
            if not export_office(illustrated, outdir, plain_md=note_path):
                log("  !! 产物不完整：docx 没生成出来（退出码 3）")
                rc = _worse(rc, RC_INCOMPLETE)
        except Exception as e:
            log(f"  !! 笔记环节失败（其余结果照常保留）: {type(e).__name__} {e}")
            log("  !! 产物不完整：笔记生成失败（退出码 3）")
            rc = _worse(rc, RC_INCOMPLETE)

    # 回看页（无额外依赖，每次都生成；出问题也不影响其它产物）
    try:
        gen_review_html(outdir, src["video"], lines_dicts, items, title,
                        source=source, ocr_on=ocr_on, duration=src["duration"])
    except Exception as e:
        log(f"  !! 回看页生成失败（其余结果照常保留）: {type(e).__name__} {e}")

    # 元信息
    with open(os.path.join(outdir, "元信息.txt"), "w", encoding="utf-8") as f:
        f.write(f"标题：{title}\n")
        f.write(f"视频文件：{video_name}\n")
        f.write(f"时长：{hhmmss(src['duration'])}（{src['duration']:.1f} 秒）\n")
        f.write(f"抽帧数：{n_frames}\n")
        f.write(f"幻灯片张数：{len(items)}\n")
        f.write(f"逐字稿段数：{len(lines)}\n")
        f.write(f"字幕来源：{source or '无'}\n")
        f.write(f"OCR：{'启用' if ocr_on else '未安装'}\n")
        f.write(f"数据源：local\n")
        if PROXY_KEY:
            f.write(f"笔记模型：{MODEL}\n")
        else:
            f.write("笔记模型：未配置（本次跳过笔记生成）\n")
        f.write(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    log("完成:", outdir)
    if rc:
        log(f"退出码 {rc}：" + {RC_NO_TRANSCRIPT: "缺逐字稿（其它产物已产出）",
                              RC_INCOMPLETE: "产物不完整（见上面 !! 提示）"}.get(rc, ""))
    return outdir, rc


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="本地录播 → 课件截图 + 图文笔记 + docx/pdf + 回看.html（不联网抓取）")
    ap.add_argument("video", help="本地视频文件")
    ap.add_argument("-t", "--transcript",
                    help="本地逐字稿：.srt / .vtt / .txt（可省略；省略则尝试可选 ASR）")
    ap.add_argument("--title", help="标题（默认取视频文件名）")
    ap.add_argument("-o", "--out", help="输出目录（默认 ./输出/<视频名>/）")
    ap.add_argument("--asr-model", default="small",
                    help="可选 ASR 的模型大小：tiny/base/small/medium/large-v3（默认 small），"
                         "也可以填本地模型目录，离线用")
    ap.add_argument("--lang", default="zh", help="可选 ASR 的语言代码（默认 zh，auto=自动判断）")
    ap.add_argument("--asr-device", default=None,
                    help="可选 ASR 的推理设备：cpu（默认）/ cuda / auto")
    args = ap.parse_args(argv)

    try:
        got = process(args.video, args.transcript, args.title, args.out,
                      asr_model=args.asr_model, lang=args.lang, asr_device=args.asr_device)
    except FileNotFoundError as e:              # 视频/逐字稿路径不存在 → 输入不可用
        log("!!", e)
        return RC_INPUT
    path, code = got if isinstance(got, tuple) else (got, RC_INPUT)
    return code if path else RC_INPUT


if __name__ == "__main__":
    sys.exit(main())
