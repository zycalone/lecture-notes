#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""可选语音识别：faster-whisper → 逐字稿 lines。

这是**可选依赖**：装不上不影响主流程（lecture_notes.py 会打一行明确指引、
以退出码 2 结束，而不是崩栈）。本模块顶层不 import faster_whisper，只有真正
要转写时才延迟导入，所以 import asr 在任何机器上都安全。

对外：

    available() -> bool          现在能不能用（只看可得性，不做转写）
    transcribe(video, model_size="small", language="zh", on_log=None) -> lines
        lines = [{"t": 秒(float), "speaker": "发言人", "text": "…"}, …]

模型体积参考（首次转写会自动下载，之后走本地缓存）：
    tiny≈75MB  base≈145MB  small≈484MB  medium≈1.5GB  large-v3≈3GB
中文讲座一般 small 就够用；想省钱可 --asr-model base，想更准用 medium。
"""
import importlib.util
import os
import sys

# huggingface_hub 拿不到权重时默认会干等 30 秒以上（etag 探测 + 下载重试）。
# 这里把超时压短，让「没联网 / 没下过权重」这种情况快点报错。
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "8")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

DEFAULT_MODEL = "small"
DEFAULT_LANG = "zh"
INSTALL_HINT = 'pip install faster-whisper（或 pip install "lecture-notes[asr]"）'
_MODULE = "faster_whisper"


class AsrUnavailable(RuntimeError):
    """没装 faster-whisper，或装上了但导入失败（缺 ctranslate2 / av 之类）。"""


class AsrWeightsUnavailable(AsrUnavailable):
    """装是装上了，但拿不到模型权重（首次使用需要联网下载，断网/离线就会走到这里）。

    单独一个异常类是为了让调用方能给「已装但下载不到权重」的提示，
    而不是误导用户去 pip install。
    """


def available():
    """能不能用（只看可得性，不做转写）。已经往 sys.modules 里塞过假模块也算可用。"""
    if _MODULE in sys.modules:
        return sys.modules.get(_MODULE) is not None
    try:
        return importlib.util.find_spec(_MODULE) is not None
    except Exception:
        return False


def _import_model_cls():
    try:
        from faster_whisper import WhisperModel
    except Exception as e:                     # ImportError / OSError / 缺 DLL 都算「装不上」
        raise AsrUnavailable(f"用不了 faster-whisper（{type(e).__name__}: {str(e)[:120]}）。{INSTALL_HINT}") from e
    return WhisperModel


DEFAULT_DEVICE = os.environ.get("SHIKE_ASR_DEVICE") or "cpu"


def _load_model(model_size, log, device=None):
    """建模型。

    默认 device=cpu：ctranslate2 的 auto 在「有 NVIDIA 驱动但没装 cuBLAS」的机器上
    会挑中 CUDA，建模型能过、一开始转写就崩（实测 cublas64_12.dll is not found）。
    想用 GPU 就传 --asr-device cuda 或设 SHIKE_ASR_DEVICE=cuda。

    建模型这一步同时会解析/下载权重，所以这里失败一律按「拿不到模型权重」上报
    （AsrWeightsUnavailable），提示文案和「没装 faster-whisper」是分开的两档。
    """
    WhisperModel = _import_model_cls()
    dev = (device or DEFAULT_DEVICE).strip().lower() or "cpu"
    # 先只看本地缓存：权重已经下好的人秒开；没下过的人也能立刻知道要联网，
    # 而不是在 huggingface_hub 里干等 30 秒才报「拿不到权重」。
    try:
        return WhisperModel(model_size, device=dev, compute_type="int8", local_files_only=True)
    except TypeError:
        pass                                   # 老版本 faster-whisper 没这个参数
    except Exception:
        log(f"    本地没有 {model_size} 权重，尝试联网下载（首次会下载）...")
    try:
        return WhisperModel(model_size, device=dev, compute_type="int8")
    except Exception as e:
        if dev != "cpu":
            log(f"    !! device={dev} 建模型失败（{type(e).__name__}: {str(e)[:60]}），改用 CPU")
            try:
                return WhisperModel(model_size, device="cpu", compute_type="int8")
            except Exception as e2:
                raise AsrWeightsUnavailable(
                    f"拿不到模型权重（{type(e2).__name__}: {str(e2)[:160]}）") from e2
        raise AsrWeightsUnavailable(
            f"拿不到模型权重（{type(e).__name__}: {str(e)[:160]}）") from e


def transcribe(video, model_size=DEFAULT_MODEL, language=DEFAULT_LANG, on_log=None,
               device=None):
    """video → lines。装不上抛 AsrUnavailable；其它失败按运行时错误抛给调用方。"""
    log = on_log or (lambda *a: None)
    lang = (language or "").strip()
    lang = None if lang.lower() in ("", "auto") else lang      # auto / 空 → 交给模型判断
    log(f"  加载 ASR 模型 faster-whisper/{model_size}（首次会下载权重）...")
    model = _load_model(model_size, log, device=device)

    log(f"  识别中（语言 {lang or 'auto'}）...  视频：{video}")
    segments, info = model.transcribe(
        video,
        language=lang,
        vad_filter=True,                       # 去掉长时间静音，讲座里很常见
        beam_size=5,
        condition_on_previous_text=False,      # 避免长音频串词/复读
        # 中文讲座里 whisper 常输出繁体，给一段简体提示词能把输出拉回简体
        initial_prompt=("以下是普通话讲座的录音，请用简体中文转写：" if (lang or "zh") == "zh" else None),
    )

    dur = float(getattr(info, "duration", 0.0) or 0.0)
    if dur:
        log(f"  音频时长 {dur:.1f} 秒，语言判定 {getattr(info, 'language', '?')}")

    lines = []
    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        try:
            t = round(float(getattr(seg, "start", 0.0) or 0.0), 2)
        except (TypeError, ValueError):
            t = 0.0
        lines.append({"t": t, "speaker": "发言人", "text": text})
        if len(lines) % 50 == 0:
            log(f"    ASR 已出 {len(lines)} 句，当前 {int(t) // 60} 分 {int(t) % 60} 秒")

    log(f"  ASR 完成：{len(lines)} 句")
    return lines


def describe(model_size=DEFAULT_MODEL):
    """写进元信息 / 逐字稿首行的「字幕来源」标签。"""
    return f"asr(faster-whisper/{model_size or DEFAULT_MODEL})"


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if len(sys.argv) < 2:
        print("用法: asr.py <video> [model_size] [lang]")
        print("可用:", available())
        raise SystemExit(1)
    print("faster-whisper 可用:", available())
    try:
        _lines = transcribe(sys.argv[1],
                            sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL,
                            sys.argv[3] if len(sys.argv) > 3 else DEFAULT_LANG,
                            on_log=print)
    except AsrUnavailable as _e:
        print("!!", _e)
        raise SystemExit(2)
    for _l in _lines[:10]:
        print(_l)
