#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""可选 OCR：课件帧 → 文字，让截图里的字可以被搜索/被笔记引用。

这是**可选依赖**（rapidocr_onnxruntime，纯 ONNX、不用装 torch）。装不上时
extract() 一律返回空串，主流程照常跑完，只是 index.json 里的文字为空。

对外：

    available() -> bool
    extract(image_path) -> str          单张图 → 文字（换行拼成一段，失败返回 ""）
    extract_lines(image_path) -> [str]  单张图 → 每行一条
    hint() -> str

延迟导入：本模块顶层不 import rapidocr_onnxruntime。
"""
import importlib.util
import os
import sys

INSTALL_HINT = 'pip install rapidocr-onnxruntime（或 pip install "lecture-notes[ocr]"）'
_MODULE = "rapidocr_onnxruntime"

_engine = None
_tried = False


def available():
    """能不能用（只看可得性，不做识别）。已经往 sys.modules 里塞过假模块也算可用。"""
    if _MODULE in sys.modules:
        return sys.modules.get(_MODULE) is not None
    try:
        return importlib.util.find_spec(_MODULE) is not None
    except Exception:
        return False


def hint():
    return f"OCR 需要可选依赖 {INSTALL_HINT}"


def _get_engine():
    """懒建引擎（onnx 模型加载慢，一个进程只建一次）。不可用返回 None。"""
    global _engine, _tried
    if _engine is not None:
        return _engine
    if _tried:
        return None
    _tried = True
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return None
    try:
        _engine = RapidOCR()
    except Exception:
        return None
    return _engine


def _rows(result):
    """把 rapidocr 各种版本的返回值（result / (result, elapse) / 嵌套 list）摊平成行列表。"""
    if result is None:
        return []
    # 新版：(result, elapse)，elapse 是数字列表
    if (isinstance(result, (tuple, list)) and len(result) == 2
            and isinstance(result[0], list) and isinstance(result[1], list)
            and (not result[1] or isinstance(result[1][0], (int, float)))):
        return result[0]
    if isinstance(result, (tuple, list)):
        return list(result)
    return []


def extract_lines(image_path):
    """单张图片 → ["第一行", "第二行", ...]。不可用 / 没字 / 出错都返回 []。"""
    if not image_path or not os.path.isfile(image_path):
        return []
    engine = _get_engine()
    if engine is None:
        return []
    try:
        raw = engine(image_path)
    except Exception:
        return []
    texts = []
    for row in _rows(raw):
        if isinstance(row, dict):
            txt = row.get("text") or row.get("txt") or ""
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            txt = row[1] if isinstance(row[1], str) else ""
        else:
            txt = ""
        txt = str(txt).strip()
        if txt:
            texts.append(txt)
    return texts


def extract(image_path, sep="　"):
    """单张图片 → 一段文字（各行用 sep 连接）。不可用 / 没字 / 出错都返回 ""。"""
    return sep.join(extract_lines(image_path))


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if len(sys.argv) < 2:
        print("用法: ocr.py <图片>")
        print("可用:", available())
        raise SystemExit(1)
    print("OCR 可用:", available())
    if not available():
        print("!!", hint())
        raise SystemExit(0)
    for _i in sys.argv[1:]:
        print(f"--- {_i}")
        print(extract(_i) or "（没识别到文字）")
