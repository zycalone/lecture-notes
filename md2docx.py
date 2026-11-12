#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Markdown → Word（含图表、课件截图；公式降级为可读 Unicode）。

可作为模块调用：build(md_path, docx_path, base_dir)
"""
import os
import re
import shutil
import subprocess
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CJK = "微软雅黑"
MONO = "Consolas"          # 行内代码 / 代码块用的等宽字体
CODE_BG = "F2F2F2"         # 代码块灰底
CODE_WRAP = 90             # 代码块长行按 90 字符软折行（Word 里不折行就会撑出页面）

SYMBOLS = [
    (r"\\mathbb\s*R", "ℝ"), (r"\\mathbb\s*N", "ℕ"), (r"\\mathbb\s*Z", "ℤ"),
    (r"\\operatorname\{arccot\}", "arccot"), (r"\\operatorname\{([a-zA-Z]+)\}", r"\1"),
    (r"\\text\{([^{}]*)\}", r"\1"), (r"\\mathrm\{([^{}]*)\}", r"\1"),
    (r"\\qquad", "    "), (r"\\quad", "  "), (r"\\,", " "), (r"\\;", " "), (r"\\!", ""),
    (r"\\left", ""), (r"\\right", ""), (r"\\displaystyle", ""),
    (r"\\Longleftrightarrow", "⇔"), (r"\\Leftrightarrow", "⇔"), (r"\\Longrightarrow", "⇒"),
    (r"\\Rightarrow", "⇒"), (r"\\iff", "⇔"), (r"\\sim", "~"), (r"\\approx", "≈"),
    (r"\\neq", "≠"), (r"\\ne\b", "≠"), (r"\\leqslant", "≤"), (r"\\geqslant", "≥"),
    (r"\\leq", "≤"), (r"\\le\b", "≤"), (r"\\geq", "≥"), (r"\\ge\b", "≥"),
    (r"\\pm", "±"), (r"\\mp", "∓"), (r"\\infty", "∞"), (r"\\to\b", "→"),
    (r"\\rightarrow", "→"), (r"\\in\b", "∈"), (r"\\notin", "∉"), (r"\\subset", "⊂"),
    (r"\\cup", "∪"), (r"\\cap", "∩"), (r"\\forall", "∀"), (r"\\exists", "∃"),
    (r"\\times", "×"), (r"\\cdot", "·"), (r"\\div", "÷"), (r"\\ast", "*"),
    (r"\\alpha", "α"), (r"\\beta", "β"), (r"\\gamma", "γ"), (r"\\delta", "δ"),
    (r"\\Delta", "Δ"), (r"\\theta", "θ"), (r"\\lambda", "λ"), (r"\\mu", "μ"),
    (r"\\pi", "π"), (r"\\varphi", "φ"), (r"\\phi", "φ"), (r"\\omega", "ω"),
    (r"\\Omega", "Ω"), (r"\\varepsilon", "ε"), (r"\\epsilon", "ε"), (r"\\sigma", "σ"),
    (r"\\sum", "Σ"), (r"\\prod", "∏"), (r"\\int", "∫"), (r"\\partial", "∂"),
    (r"\\cdots", "…"), (r"\\ldots", "…"), (r"\\dots", "…"),
    (r"\\sin", "sin"), (r"\\cos", "cos"), (r"\\tan", "tan"), (r"\\cot", "cot"),
    (r"\\sec", "sec"), (r"\\csc", "csc"), (r"\\arcsin", "arcsin"), (r"\\arccos", "arccos"),
    (r"\\arctan", "arctan"), (r"\\arccot", "arccot"), (r"\\ln", "ln"), (r"\\log", "log"),
    (r"\\exp", "exp"), (r"\\lim", "lim"), (r"\\max", "max"), (r"\\min", "min"),
    (r"\\sqrt", "√"), (r"\\%", "%"), (r"\\\{", "{"), (r"\\\}", "}"), (r"\\\\\\\\",
     "；"),
]

SUP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")


def _frac(m):
    a, b = m.group(1), m.group(2)
    if re.search(r"[+\-− ]", a):
        a = f"({a})"
    if re.search(r"[+\-− ]", b):
        b = f"({b})"
    return f"{a}/{b}"


def latex_to_text(src):
    s = src
    for _ in range(6):
        new = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", _frac, s)
        new = re.sub(r"\\frac\s*(\d)\s*(\d)", lambda m: f"{m.group(1)}/{m.group(2)}", new)
        if new == s:
            break
        s = new
    s = re.sub(r"\\sqrt\{([^{}]*)\}", lambda m: "√(" + m.group(1) + ")", s)
    s = re.sub(r"\\sqrt\s*(\w)", lambda m: "√" + m.group(1), s)
    for pat, rep in SYMBOLS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\^\{([^{}]*)\}", lambda m: m.group(1).translate(SUP)
               if re.fullmatch(r"[0-9+\-=()n]+", m.group(1)) else "^(" + m.group(1) + ")", s)
    s = re.sub(r"_\{([^{}]*)\}", lambda m: "_(" + m.group(1) + ")", s)
    s = re.sub(r"\^(\w)", lambda m: m.group(1).translate(SUP)
               if m.group(1).isdigit() else "^" + m.group(1), s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"[ \t]{2,}", " ", s).strip()


def conv_inline(text):
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: latex_to_text(m.group(1)), text, flags=re.S)
    text = re.sub(r"\$(.+?)\$", lambda m: latex_to_text(m.group(1)), text, flags=re.S)
    return text


def add_runs(par, text, italic=False):
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not part:
            continue
        is_code = False
        if part.startswith("**") and part.endswith("**"):
            r = par.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = MONO
            r.font.color.rgb = RGBColor(0xB0, 0x30, 0x60)
            is_code = True
        else:
            r = par.add_run(part)
        r.italic = italic
        if is_code:
            # 行内代码保留 Consolas：不能再写中文字体，否则 font.name 被覆盖回微软雅黑
            try:
                r._element.rPr.rFonts.set(qn("w:eastAsia"), MONO)
            except Exception:
                pass
            continue
        r.font.name = CJK
        try:
            r._element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
        except Exception:
            pass


def _shade(par, fill=CODE_BG):
    """给段落加一层灰底（w:shd），代码块靠它看起来像一块。"""
    try:
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), fill)
        par._p.get_or_add_pPr().append(shd)
    except Exception:
        pass


def add_code_block(doc, body, lang=""):
    """``` 围栏里的内容 → 等宽 Consolas + 灰底的一段段代码行。

    代码块内不做任何行内 Markdown 转换（** / ` 原样保留），长行按 CODE_WRAP 字符软折行。
    有语言名就先给一个小标签（浅灰小字），方便阅读时知道是什么语言。
    """
    if lang:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(lang)
        r.font.name = MONO
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
        _shade(p)
    rows = body or [""]
    for k, line in enumerate(rows):
        chunks = [line[i:i + CODE_WRAP] for i in range(0, len(line), CODE_WRAP)] or [""]
        for ci, chunk in enumerate(chunks):
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.space_before = Pt(6 if (k == 0 and ci == 0 and not lang) else 0)
            pf.space_after = Pt(6 if k == len(rows) - 1 else 0)
            pf.line_spacing = 1.0
            pf.left_indent = Cm(0.2)
            r = p.add_run(chunk if chunk else " ")
            r.font.name = MONO
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
            try:
                r._element.rPr.rFonts.set(qn("w:eastAsia"), MONO)
            except Exception:
                pass
            _shade(p)


def _styles(doc):
    st = doc.styles["Normal"]
    st.font.name = CJK
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    st.paragraph_format.space_after = Pt(4)
    st.paragraph_format.line_spacing = 1.25
    for name, size in (("Heading 1", 18), ("Heading 2", 15), ("Heading 3", 12.5)):
        try:
            h = doc.styles[name]
            h.font.name = CJK
            h.font.size = Pt(size)
            h.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
            h.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
        except Exception:
            pass


def build(md_path, docx_path, base_dir=None, with_images=True):
    base_dir = base_dir or os.path.dirname(os.path.abspath(md_path))
    md = open(md_path, encoding="utf-8-sig").read()  # utf-8-sig：带 BOM 的 md 也能正确识别首行标题
    doc = Document()
    _styles(doc)
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    usable = Cm(21.0 - 4.4)

    lines = md.splitlines()
    i, img_count = 0, 0
    bad_images = []
    while i < len(lines):
        s = lines[i].strip()

        if not s or s == "---":
            i += 1
            continue

        # ``` 围栏代码块：整块收走，等宽 + 灰底，内部不解析行内 Markdown
        m = re.match(r"^```+\s*([^`\s]*)\s*$", s)
        if m:
            lang = m.group(1)
            i += 1
            body = []
            while i < len(lines) and not re.match(r"^```+\s*$", lines[i].strip()):
                body.append(lines[i].rstrip())
                i += 1
            i += 1                                  # 跳过收尾的 ```
            if not body:
                body = [""]
            add_code_block(doc, body, lang)
            continue

        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[\s:|-]+$", "".join(cells)):
                    rows.append(cells)
                i += 1
            if rows:
                ncol = max(len(r) for r in rows)
                tbl = doc.add_table(rows=0, cols=ncol)
                tbl.style = "Table Grid"
                for ri, row in enumerate(rows):
                    cells = tbl.add_row().cells
                    for ci in range(ncol):
                        p = cells[ci].paragraphs[0]
                        p.paragraph_format.space_after = Pt(1)
                        add_runs(p, conv_inline(row[ci]) if ci < len(row) else "")
                        if ri == 0:
                            for r in p.runs:
                                r.bold = True
                doc.add_paragraph()
            continue

        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
        if m:
            alt, src_rel = m.group(1), m.group(2)
            if with_images:
                path = os.path.join(base_dir, src_rel.replace("/", os.sep))
                if os.path.exists(path):
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.paragraph_format.space_before = Pt(6)
                    try:
                        p.add_run().add_picture(path, width=int(usable * 0.92))
                        img_count += 1
                    except Exception as e:
                        # 损坏/0 字节/格式不认的图片：python-docx 会抛
                        # UnrecognizedImageError（str 还是空的），不能让它毁掉整份 docx
                        why = str(e).strip() or type(e).__name__
                        label = alt or src_rel
                        print(f"  图片无法读取: {src_rel}（{why}）→ 插入文字占位")
                        p._p.getparent().remove(p._p)
                        ph = doc.add_paragraph()
                        ph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        r = ph.add_run(f"[图片无法读取：{label}]")
                        r.font.color.rgb = RGBColor(0xC0, 0x30, 0x30)
                        r.italic = True
                        bad_images.append(src_rel)
                else:
                    print("  缺图:", path)
            i += 1
            continue

        if s.startswith("*") and s.endswith("*") and not s.startswith("**"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(10)
            add_runs(p, conv_inline(s.strip("*")), italic=True)
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lvl = len(m.group(1))
            h = doc.add_heading("", level=0 if lvl == 1 else min(lvl - 1, 3))
            h.paragraph_format.space_before = Pt(14 if lvl <= 2 else 10)
            h.paragraph_format.space_after = Pt(6)
            add_runs(h, conv_inline(m.group(2).strip()))
            i += 1
            continue

        if s.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            p.paragraph_format.space_after = Pt(2)
            add_runs(p, conv_inline(s.lstrip("> ").strip()), italic=True)
            for r in p.runs:
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            i += 1
            continue

        m = re.match(r"^[-*]\s+(.*)$", s)
        if m:
            add_runs(doc.add_paragraph(style="List Bullet"), conv_inline(m.group(1)))
            i += 1
            continue
        m = re.match(r"^(\d+)[.)]\s+(.*)$", s)
        if m:
            add_runs(doc.add_paragraph(style="List Number"), conv_inline(m.group(2)))
            i += 1
            continue

        add_runs(doc.add_paragraph(), conv_inline(s))
        i += 1

    doc.save(docx_path)
    print(f"  docx: {len(doc.paragraphs)} 段, {len(doc.tables)} 表, {img_count} 图"
          + (f"，{len(bad_images)} 张图片读不出来（已插占位文字）："
             + "、".join(bad_images) if bad_images else ""))
    return docx_path


def _find_libreoffice():
    """返回 (node, cli) 或 ("", soffice.exe 路径)，找不到返回 None。"""
    node = os.environ.get("SHIKE_NODE") or (shutil.which("node") or "")
    cli = os.environ.get("SHIKE_LO_CLI") or ""
    if os.path.isfile(node) and os.path.isfile(cli):
        return ("kit", node, cli)
    for p in (r"C:\Program Files\LibreOffice\program\soffice.exe",
              r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
              r"D:\Program Files\LibreOffice\program\soffice.exe"):
        if os.path.isfile(p):
            return ("soffice", p, "")
    return None


def to_pdf(docx_path, pdf_path):
    """docx -> pdf，失败返回 None，不抛异常。"""
    lo = _find_libreoffice()
    if not lo:
        print("  （没找到 LibreOffice，跳过 pdf；装 LibreOffice 后会顺带出 pdf，"
              "或直接用 Word 打开 docx 另存为 PDF）")
        return None
    try:
        # LibreOffice 要在系统 TEMP 下建临时目录，本机 TEMP 常被策略挡（EPERM），统一指到脚本旁的 _lo_tmp
        lo_tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_lo_tmp")
        os.makedirs(lo_tmp, exist_ok=True)
        env = dict(os.environ, TEMP=lo_tmp, TMP=lo_tmp, TMPDIR=lo_tmp)
        if lo[0] == "kit":
            r = subprocess.run([lo[1], lo[2], "convert", "--input", docx_path,
                                "--output", pdf_path], capture_output=True, env=env)
        else:
            outdir = os.path.dirname(os.path.abspath(pdf_path))
            r = subprocess.run([lo[1], "--headless", "--convert-to", "pdf",
                                "--outdir", outdir, docx_path], capture_output=True, env=env)
            made = os.path.join(outdir, os.path.splitext(os.path.basename(docx_path))[0] + ".pdf")
            if os.path.isfile(made) and os.path.abspath(made) != os.path.abspath(pdf_path):
                os.replace(made, pdf_path)
    except Exception as e:
        print("  pdf 转换异常:", e)
        return None
    if os.path.isfile(pdf_path):
        return pdf_path
    err = (r.stderr or r.stdout or b"")[:300].decode("utf-8", "replace").strip()
    print("  pdf 转换失败:", err or f"退出码 {r.returncode}")
    return None


def export(md_path, outdir=None, stem="笔记", pdf=True):
    """md -> docx(+pdf)，名字取 stem。返回生成的文件名列表（不含路径）。

    解析类脚本收尾统一调这个，保证「有 md 就有文档」。
    """
    md_path = os.path.abspath(md_path)
    outdir = os.path.abspath(outdir or os.path.dirname(md_path))
    os.makedirs(outdir, exist_ok=True)
    docx_path = os.path.join(outdir, stem + ".docx")
    made = []
    try:
        build(md_path, docx_path, base_dir=os.path.dirname(md_path))
        made.append(os.path.basename(docx_path))
    except Exception as e:
        # 不能只打印 str(e)：UnrecognizedImageError 之类异常的消息是空的，日志会一片空白
        print(f"  docx 生成失败: {type(e).__name__} {str(e).strip() or '（没有错误信息）'}")
        return made
    if pdf:
        pdf_path = os.path.join(outdir, stem + ".pdf")
        if to_pdf(docx_path, pdf_path):
            made.append(os.path.basename(pdf_path))
    return made


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: md2docx.py <input.md> [输出目录] [文件名前缀]")
        print("      md2docx.py 笔记.md            → 笔记.md 同目录生成 笔记.docx / 笔记.pdf")
        sys.exit(1)
    src = sys.argv[1]
    od = sys.argv[2] if len(sys.argv) > 2 else None
    st = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(os.path.basename(src))[0]
    got = export(src, od, st)
    print("生成:", ", ".join(got) if got else "（无）")
    sys.exit(0 if got else 1)
