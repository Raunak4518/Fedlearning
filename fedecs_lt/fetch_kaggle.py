"""Downloads the outputs of finished Kaggle notebooks and files them under fedecs_lt/results/<notebook>/ (repository
tool). Public notebooks need no Kaggle login: the listing API answers without one.

    python fedecs_lt/fetch_kaggle.py USER/SLUG [USER/SLUG ...]          # or full https://www.kaggle.com/code/... links
    python fedecs_lt/fetch_kaggle.py --all                               # every notebook in results/KAGGLE_SOURCES.md
    python fedecs_lt/fetch_kaggle.py --results USER/SLUG                 # also the large NAME__results.zip

For each notebook it keeps, in the repository, the small text files that every table needs (runs.jsonl, runs.csv,
failures.jsonl, audit.json, report_info.json, code_version.txt) and writes summary.md (runs finished against the
plan, failures grouped by error, the report's checks). The report PDF and the zips are downloaded to
fedecs_lt/results/_downloads/ (ignored by git: too large for the repository; Kaggle keeps them). The notebook's
Kaggle address is recorded in results/KAGGLE_SOURCES.md so that everything can be downloaded again.
"""
import collections
import datetime
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DL = os.path.join(RES, "_downloads")
SOURCES = os.path.join(RES, "KAGGLE_SOURCES.md")
KEEP = ("runs.jsonl", "runs.csv", "failures.jsonl", "audit.json", "report_info.json", "code_version.txt")
API = "https://www.kaggle.com/api/v1/kernels/output?"


def ref_of(arg):
    m = re.search(r"kaggle\.com/code/([^/]+)/([^/?#]+)", arg)
    return (m.group(1), m.group(2)) if m else tuple(arg.strip("/").split("/")[-2:])


def listing(user, slug):
    files, tok = [], None
    while True:
        q = dict(userName=user, kernelSlug=slug, **({"pageToken": tok} if tok else {}))
        with urllib.request.urlopen(API + urllib.parse.urlencode(q), timeout=120) as r:
            d = json.load(r)
        files += d.get("files", [])
        tok = d.get("nextPageToken")
        if not tok:
            return files


def planned_runs():
    sys.path.insert(0, HERE)
    import build_notebooks
    return {n: len(nb["runs"]) for n, nb in build_notebooks.all_notebooks().items()}


def summary_md(name, url, d, plan):
    runs = [json.loads(x) for x in open(os.path.join(d, "runs.jsonl")) if x.strip()] \
        if os.path.exists(os.path.join(d, "runs.jsonl")) else []
    fails = [json.loads(x) for x in open(os.path.join(d, "failures.jsonl")) if x.strip()] \
        if os.path.exists(os.path.join(d, "failures.jsonl")) else []
    done = {r["run_id"] for r in runs}
    failed = sorted({f["run_id"] for f in fails} - done)       # failed on every attempt
    info = json.load(open(os.path.join(d, "report_info.json"))) if os.path.exists(os.path.join(d, "report_info.json")) \
        else {}
    version = open(os.path.join(d, "code_version.txt")).read().strip() if os.path.exists(
        os.path.join(d, "code_version.txt")) else "not recorded"
    n_plan = plan.get(name)
    complete = n_plan is not None and len(done) == n_plan
    lines = [f"# {name}: {'complete' if complete else 'INCOMPLETE'}", "",
             f"Kaggle: {url}  ", f"Downloaded: {datetime.date.today().isoformat()}  ", f"Code: {version}", "",
             f"Runs finished: {len(done)} of {n_plan if n_plan is not None else '?'} planned. "
             f"Runs that failed on every attempt: {len(failed)}.", ""]
    if info:
        lines += ["Report checks: " + ", ".join(f"{k} {info[k]}" for k in ("failures", "figure_errors", "table_errors",
                                                                             "layout_problems", "caption_problems",
                                                                             "audit_problems") if k in info), ""]
    if failed:
        errs = collections.Counter()
        last = {}
        for f in fails:
            last[f["run_id"]] = f["error"]
        for rid in failed:
            e = last[rid]
            key = ("Ray out of memory (host RAM)" if "OutOfMemoryError" in e and "running low on memory" in e else
                   "CUDA out of memory (GPU)" if "CUDA out of memory" in e or "OutOfMemoryError: CUDA" in e else
                   "download failed (connection timed out)" if "URLError" in e else e[-160:])
            errs[key] += 1
        lines += ["Failed runs by cause:", ""] + [f"- {k}: {v}" for k, v in errs.most_common()] + ["", "Failed runs:", ""]
        lines += [f"- {rid}" for rid in failed] + [""]
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    return complete, len(done), n_plan, len(failed)


def record_source(name, url):
    rows = open(SOURCES).read().splitlines() if os.path.exists(SOURCES) else [
        "# Kaggle notebooks whose outputs are in this folder", "",
        "Download again with `python fedecs_lt/fetch_kaggle.py --all`.", "", "| Notebook | Kaggle |", "|---|---|"]
    rows = [r for r in rows if not r.startswith(f"| {name} |")] + [f"| {name} | {url} |"]
    head = [r for r in rows if not r.startswith("| NB")]
    body = sorted(r for r in rows if r.startswith("| NB"))
    with open(SOURCES, "w") as f:
        f.write("\n".join(head + body) + "\n")


def fetch(user, slug, plan, results=False):
    url = f"https://www.kaggle.com/code/{user}/{slug}"
    files = [f for f in listing(user, slug) if "/" not in f["fileName"]]
    names = {f["fileName"]: f["url"] for f in files}
    summ = [n for n in names if n.endswith("__summary.zip")]
    if not summ:
        print(f"{url}: no NAME__summary.zip in its output (still running, or not saved as a version)")
        return None
    name = summ[0][: -len("__summary.zip")]
    os.makedirs(DL, exist_ok=True)
    want = [summ[0], name + "__report.pdf"] + ([name + "__results.zip"] if results else [])
    for fn in want:
        if fn in names:
            urllib.request.urlretrieve(names[fn], os.path.join(DL, fn))
    d = os.path.join(RES, name)
    os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(os.path.join(DL, summ[0])) as z:     # only named files are read: nothing else is unpacked
        for member in z.namelist():
            base = os.path.basename(member)
            if base in KEEP and member.count("/") <= 1:
                with open(os.path.join(d, base), "wb") as f:
                    f.write(z.read(member))
    record_source(name, url)
    complete, n, n_plan, n_failed = summary_md(name, url, d, plan)
    print(f"{name}: {n} of {n_plan} runs{'' if complete else ' -- INCOMPLETE'}"
          f"{f', {n_failed} failed' if n_failed else ''}  ({url})")
    return name


def main():
    args = sys.argv[1:]
    results = "--results" in args
    args = [a for a in args if a != "--results"]
    if args == ["--all"]:
        args = re.findall(r"https://www\.kaggle\.com/code/\S+", open(SOURCES).read()) if os.path.exists(SOURCES) else []
    if not args:
        raise SystemExit(__doc__)
    plan = planned_runs()
    for a in args:
        fetch(*ref_of(a), plan, results=results)
    import gefl_measure   # GeFL's measure (every model's best round) from the per-round logs just downloaded
    gefl_measure.main()


if __name__ == "__main__":
    main()
