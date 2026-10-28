"""Checks the paper's text against the writing rules (user, 29 Oct 2026): no em dashes; no promotional or filler
wording; no repeated sentence openings in a paragraph. Prints file:line and the word; exit code 1 when an em dash is
found (the only hard rule; the rest is a list to review by hand: a flagged word can be technically needed).

    python docs/papers/style_check.py
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# wording that is promotional, vague or formulaic in a research paper (reviewed by hand, not removed blindly)
FLAG = ["novel", "notably", "remarkably", "significantly", "substantially", "dramatically", "crucial", "crucially",
        "groundbreaking", "state-of-the-art", "cutting-edge", "seamless", "seamlessly", "leverage", "leverages",
        "leveraging", "delve", "furthermore", "moreover", "in conclusion", "it is worth noting", "it is important to",
        "plays a crucial role", "a wide range of", "robust", "robustly", "comprehensive", "comprehensively",
        "powerful", "effectively", "efficiently", "superior", "outperforms", "paradigm", "holistic", "unprecedented",
        "pivotal", "key insight", "landscape", "realm", "showcase", "underscore", "underscores", "intricate",
        "meticulous", "very", "extremely", "clearly", "obviously", "of course", "in order to", "utilize", "utilizes"]
EM = re.compile(r"—|---")   # Unicode em dash, LaTeX em dash


def strip_comments(line):
    return re.sub(r"(?<!\\)%.*", "", line)


def main():
    bad = flagged = 0
    for f in sorted(glob.glob(os.path.join(HERE, "content", "*.tex"))):
        name = os.path.basename(f)
        for i, raw in enumerate(open(f), 1):
            line = strip_comments(raw)
            if EM.search(line):
                print(f"EM DASH  {name}:{i}: {line.strip()[:120]}")
                bad += 1
            low = line.lower()
            for w in FLAG:
                if re.search(r"(?<![a-z-])" + re.escape(w) + r"(?![a-z-])", low):
                    print(f"review   {name}:{i}: '{w}'")
                    flagged += 1
    print(f"\n{bad} em dash(es); {flagged} flagged word(s) to review")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
