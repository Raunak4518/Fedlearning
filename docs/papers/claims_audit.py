"""Lists every \\claim{...} of the paper with its location, the result slots it uses and their filled values, for the
claim-to-evidence audit (docs/papers/review/claims.md). The evidence and status columns are kept from the existing
claims.md when a claim's text is unchanged (matched by its first 60 characters), so the audit survives edits.

    python docs/papers/claims_audit.py            # rewrites review/claims.md
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "review", "claims.md")
ORDER = ["abstract", "sec1_intro", "sec2_related", "sec3_setting", "sec4_analysis", "sec5_method", "sec6_setup",
         "sec7_results", "sec8_discussion", "appendix"]


def braced(text, i):
    """The content of the {...} group starting at text[i] == '{' and the index after it."""
    depth, j = 0, i
    while j < len(text):
        if text[j] == "\\":
            j += 2
            continue
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1
    return text[i + 1:], len(text)


def numbers():
    out = {}
    p = os.path.join(HERE, "generated", "numbers.tex")
    if os.path.exists(p):
        for line in open(p):
            m = re.match(r"\\defnum\{(.+?)\}\{(.*)\}\s*$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def claims():
    nums = numbers()
    for name in ORDER:
        f = os.path.join(HERE, "content", name + ".tex")
        if not os.path.exists(f):
            continue
        text = open(f).read()
        for m in re.finditer(r"\\claim\{", text):
            body, _ = braced(text, m.end() - 1)
            line = text.count("\n", 0, m.start()) + 1
            slots = re.findall(r"\\R\{([^}]*)\}", body)
            filled = re.sub(r"\\R\{([^}]*)\}", lambda s: "«" + nums.get(s.group(1), "MISSING:" + s.group(1)) + "»", body)
            plain = re.sub(r"\s+", " ", re.sub(r"\\(cref|Cref|cite)\{[^}]*\}", "[ref]", filled)).strip()
            yield dict(where=f"{name}.tex:{line}", slots=slots, text=plain)


def previous():
    """Evidence and status of the claims in the current claims.md, keyed by the claim's first 60 characters."""
    keep = {}
    if not os.path.exists(OUT):
        return keep
    for line in open(OUT):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0].startswith("C") and cells[0][1:].isdigit():
            keep[cells[2][:60]] = (cells[4], cells[5])
    return keep


def main():
    keep = previous()
    rows = ["# Claim-to-evidence audit", "",
            "Every `\\claim{}` of the paper (written by `python docs/papers/claims_audit.py`; slot values in «»). "
            "Status: **Supported** (direct evidence, stated scope), **Partially supported** (evidence narrower than the "
            "wording), **Unsupported**, **Unresolved** (depends on a pending experiment). The evidence and status "
            "columns are filled by the reviewers and kept across rewrites while a claim's text is unchanged.", "",
            "| ID | Where | Claim (filled) | Slots | Evidence | Status |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(claims(), 1):
        ev, st = keep.get(c["text"].replace("|", "/")[:60], ("", ""))
        rows.append(f"| C{i} | {c['where']} | {c['text'].replace('|', '/')} | {', '.join(c['slots']) or '-'} | "
                    f"{ev} | {st} |")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(rows) + "\n")
    print(f"{len(rows) - 6} claims -> {OUT}")


if __name__ == "__main__":
    main()
