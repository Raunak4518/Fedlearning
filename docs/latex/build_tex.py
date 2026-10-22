"""Builds the LaTeX version of the paper from docs/paper_draft.md (the single source; edit the Markdown, not the .tex).

    python docs/latex/build_tex.py            -> docs/latex/paper.tex
    python docs/latex/build_tex.py --pdf      -> also docs/latex/paper.pdf (needs pdflatex)

Needs pandoc. Section numbers stay as written in the draft (LaTeX numbering is off), so "§6.11" in the text still
matches. Numeric citations [n], [n, m], [n–m] outside math become \\cite{rN}; the reference list becomes a
thebibliography. Entries marked [PENDING: ...] are kept and highlighted.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "paper_draft.md"))
OUT = os.path.join(HERE, "paper.tex")

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.5cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath,amssymb,amsthm,mathtools,bm}
\usepackage{booktabs,longtable,array,calc,multirow}
\usepackage{etoolbox}
\usepackage{xcolor}
\usepackage{newunicodechar}
\PassOptionsToPackage{hyphens}{url}
\usepackage[hidelinks]{hyperref}
\usepackage[nameinlink]{cleveref}
\setcounter{secnumdepth}{0}   % section numbers are part of the titles, as in the draft
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\AtBeginEnvironment{longtable}{\small}
\renewcommand{\arraystretch}{1.15}
\newcommand{\pending}[1]{\textcolor{red!70!black}{\textbf{[PENDING: #1]}}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{∞}{\ensuremath{\infty}}
\newunicodechar{ε}{\ensuremath{\varepsilon}}
\newunicodechar{τ}{\ensuremath{\tau}}
\newunicodechar{κ}{\ensuremath{\kappa}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{…}{\ldots}
\newunicodechar{ı}{\i}
"""


def protect_math(s):
    """Replace $...$ / $$...$$ spans by placeholders so citation rewriting never touches math."""
    spans = []

    def keep(m):
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"
    s = re.sub(r"\$\$.*?\$\$", keep, s, flags=re.S)
    s = re.sub(r"(?<![\\$])\$(?!\$)(?:\\.|[^$\\])+?\$", keep, s)
    s = re.sub(r"`[^`\n]*`", keep, s)
    return s, spans


def restore(s, spans):
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], s)


def cite_keys(body, nrefs):
    def expand(tok):
        tok = tok.strip()
        m = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", tok)
        return list(range(int(m.group(1)), int(m.group(2)) + 1)) if m else [int(tok)]

    def sub(m):
        nums = [n for t in m.group(1).split(",") for n in expand(t)]
        if not all(1 <= n <= nrefs for n in nums):
            return m.group(0)
        return "\\cite{" + ",".join(f"r{n}" for n in nums) + "}"
    return re.sub(r"(?<![\w\]])\[(\d+(?:\s*[–-]\s*\d+)?(?:\s*,\s*\d+(?:\s*[–-]\s*\d+)?)*)\](?![(\[])", sub, body)


def pandoc(md):
    r = subprocess.run(["pandoc", "-f", "markdown+tex_math_dollars+pipe_tables-implicit_figures-auto_identifiers",
                        "-t", "latex", "--wrap=preserve", "--columns=72"],
                       input=md, capture_output=True, text=True, check=True)
    return r.stdout


def blank_before_lists(md):
    """Pandoc needs a blank line before a list; the draft (GitHub style) often has none."""
    out, prev = [], ""
    item = re.compile(r"^(?:[*-]|\d+\.) \S")
    for line in md.split("\n"):
        if item.match(line) and prev.strip() and not item.match(prev) and not prev.startswith((" ", "|", "\t")):
            out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def autolink(md):
    """Bare URLs -> <url>, so pandoc emits breakable \\url{}."""
    return re.sub(r"(?<![<(])(https?://[^\s<>)]+[^\s<>).,;])", r"<\1>", md)


def breakable_paths(tex):
    """\\texttt{path/with\\_underscores} -> \\nolinkurl{...}, which can break at / and _ ."""
    def sub(m):
        raw = m.group(1).replace("\\_", "_")
        return "\\nolinkurl{" + raw + "}" if re.fullmatch(r"[\w./*:-]+", raw) and ("/" in raw or "_" in raw) else m.group(0)
    return re.sub(r"\\texttt\{((?:[^{}\\]|\\_)+)\}", sub, tex)


def bib_entry_tex(text):
    return pandoc(autolink(text)).strip()


def main():
    md = open(SRC, encoding="utf-8").read()
    lines = md.split("\n")
    title = lines[0].lstrip("# ").strip()
    md = "\n".join(lines[1:])

    # working-draft note (first italic paragraph) -> LaTeX comment
    note, md = re.match(r"\s*(\*Working draft.*?\*)\s*\n(.*)", md, flags=re.S).groups()

    abstract, rest = re.match(r"\s*-{3,}\s*\n## Abstract\s*\n(.*?)\n-{3,}\s*\n(.*)", md, flags=re.S).groups()
    body, refs = re.match(r"(.*)\n## References\s*\n(.*)", rest, flags=re.S).groups()

    # reference list: numbered entries [n] ...; unnumbered bullets (cited by author-year) first
    unnum = [m.strip() for m in re.findall(r"^\* (.*?)(?=^\* |^\[\d+\]|\Z)", refs, flags=re.S | re.M)]
    numbered = re.findall(r"^\[(\d+)\]\s*(.*?)(?=^\[\d+\]|\Z)", refs, flags=re.S | re.M)
    nrefs = max(int(n) for n, _ in numbered)

    body = blank_before_lists(re.sub(r"^-{3,}\s*$", "", body, flags=re.M))                       # horizontal rules
    body = re.sub(r"\*\*\[PENDING: ([^\]]*)\]\*\*", r"\\pending{\1}", body)   # highlighted, raw LaTeX
    body, spans = protect_math(body)
    abstract, spans_a = protect_math(abstract)
    body = restore(cite_keys(body, nrefs), spans)
    abstract = restore(cite_keys(abstract, nrefs), spans_a)

    tex_body = breakable_paths(pandoc(autolink(body)))
    tex_abs = pandoc(abstract)
    bib = ["\\begin{thebibliography}{99}", "\\small"]
    for i, e in enumerate(unnum):
        bib.append(f"\\bibitem{{u{i + 1}}} " + bib_entry_tex(" ".join(e.split())))
    for n, e in numbered:
        bib.append(f"\\bibitem{{r{n}}} " + bib_entry_tex(" ".join(e.split())))
    bib.append("\\end{thebibliography}")

    note_tex = "\n".join("% " + l for l in note.strip("*").split("\n"))
    title_tex = pandoc(title).strip()
    out = "\n".join([
        "% GENERATED by docs/latex/build_tex.py from docs/paper_draft.md. Do not edit by hand.",
        note_tex, PREAMBLE,
        f"\\title{{{title_tex}}}", "\\author{Anonymous authors}", "\\date{}",
        "\\begin{document}", "\\maketitle",
        "\\begin{abstract}", tex_abs.strip(), "\\end{abstract}",
        "\\tableofcontents", "\\bigskip",
        tex_body.strip(),
        "\\nocite{" + ",".join(f"u{i + 1}" for i in range(len(unnum))) + "}" if unnum else "",
        "\n".join(bib),
        "\\end{document}", ""])
    open(OUT, "w", encoding="utf-8").write(out)
    print(f"wrote {OUT}: {len(numbered)} numbered references, {len(unnum)} unnumbered, "
          f"{out.count(chr(92) + 'cite{')} citations")
    if "--pdf" in sys.argv:
        for _ in range(3):  # bibliography and table of contents need repeated passes
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"],
                               cwd=HERE, capture_output=True, text=True)
            if r.returncode:
                print(r.stdout[-3000:])
                sys.exit(1)
        for ext in ("aux", "log", "out", "toc"):
            p = os.path.join(HERE, "paper." + ext)
            if os.path.exists(p):
                os.remove(p)
        print("wrote", os.path.join(HERE, "paper.pdf"))


if __name__ == "__main__":
    main()
