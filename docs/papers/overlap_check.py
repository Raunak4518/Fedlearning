"""Originality check without an external service: word n-gram overlap between the paper's text and the text of the
cited sources (plain-text copies in a folder, one file per source), and repeated passages inside the paper.

    python docs/papers/overlap_check.py SOURCES_DIR [--n 8]

Reports every run of at least n identical words (after lower-casing and removing LaTeX commands and punctuation)
shared with a source, with the paper location, and every run of 12 words that occurs twice in the paper. Short
technical phrases (dataset and method names, standard terms) produce some matches; each one is to be read by hand:
copied wording is rewritten or quoted and cited, a standard term is kept.
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["abstract", "sec1_intro", "sec2_related", "sec3_setting", "sec4_analysis", "sec5_method", "sec6_setup",
         "sec7_results", "sec8_discussion", "appendix"]


def clean(text):
    text = re.sub(r"(?<!\\)%.*", " ", text)                  # comments
    text = re.sub(r"\$[^$]*\$", " ", text)                    # inline maths
    text = re.sub(r"\\(begin|end)\{[^}]*\}", " ", text)
    text = re.sub(r"\\(cite|cref|Cref|ref|label|R|slottab|slotfig|whenfig|whentab|whennum)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)               # other commands (their arguments stay as words)
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def paper_words():
    out = []   # (word, file, line)
    for name in FILES:
        p = os.path.join(HERE, "content", name + ".tex")
        for i, line in enumerate(open(p), 1):
            if "\\phrow" in line or re.match(r"\s*(Method|Dataset|Setting|Question) &", line) or "tabular" in line:
                continue   # placeholder rows and headers of generated tables (never printed as such)
            out += [(w, name, i) for w in clean(line)]
    return out


def grams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def main():
    src_dir = sys.argv[1]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 8
    P = paper_words()
    pw = [w for w, _, _ in P]
    hits = 0
    for f in sorted(glob.glob(os.path.join(src_dir, "*.txt"))):
        sw = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", open(f, errors="ignore").read().lower())
        G = grams(sw, n)
        i = 0
        while i <= len(pw) - n:
            if tuple(pw[i:i + n]) in G:
                j = i + n
                while j < len(pw) and tuple(pw[j - n + 1:j + 1]) in G:
                    j += 1
                print(f"{os.path.basename(f)[:-4]:28s} {P[i][1]}.tex:{P[i][2]}  ({j - i} words) {' '.join(pw[i:j])}")
                hits += 1
                i = j
            else:
                i += 1
    print(f"\n{hits} passage(s) of >= {n} words shared with a source")
    seen, rep, last = {}, 0, -99
    for i in range(len(pw) - 11):
        g = tuple(pw[i:i + 12])
        if g in seen and P[seen[g]][1:] != P[i][1:] and i - seen[g] > 12:
            if i - last > 12:   # one report per repeated passage, not per overlapping 12-gram
                print(f"repeated  {P[seen[g]][1]}.tex:{P[seen[g]][2]} and {P[i][1]}.tex:{P[i][2]}: {' '.join(g)}")
                rep += 1
            last = i
        seen.setdefault(g, i)
    print(f"{rep} repeated 12-word passage(s) inside the paper")


if __name__ == "__main__":
    main()
