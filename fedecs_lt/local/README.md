# Running the notebooks on your own Windows PC with an NVIDIA GPU (RTX 3050 4 GB to RTX 4060 8 GB)

This runs the same experiments as the Kaggle notebooks, on your own GPU. It uses the same code (`fedecs_lt/src`) and
the same client splits and seeds, and it writes the same four output files. Results join the Kaggle results as they
are: same splits and seeds = valid pairing.

Each job has the name of a Kaggle notebook. Jobs that already have results in `fedecs_lt/results/` are hidden.

No Linux and no administrator rights are needed. Everything installs for your Windows user only.

How it differs from Kaggle:
- **One run at a time.** Kaggle runs four at a time on two T4 GPUs, so a job takes about 2-4 times its Kaggle hours.
- **No CUDA graphs** on GPUs under 12 GB. The computation is the same; the run is a little slower and holds less
  memory.
- **No 12-hour limit.** You can stop a job and resume it.
- **The one-shared-model jobs (NB20, NB29) stay on Kaggle.** They run the authors' LongTailLib code, which needs git
  and has never run on Windows.

Runs on a GPU, Kaggle or local, are not bit-identical from one run to the next. A second Kaggle run of the BloodMNIST
notebook moved method means by at most 2.5 points (`fedecs_lt/results/_repeats/README.md`).

**Native Windows has not been tested yet:** the code was developed on Linux (Kaggle). The first `run_check.bat` on
each PC is the test. If it reports anything other than FITS / DOES NOT FIT, send the window's text and the folder it
names.

## What you need
- An NVIDIA driver, **version 560 or newer**. Check with `nvidia-smi` in a Command Prompt; update from nvidia.com
  (Game Ready or Studio driver) if it is older.
- About **25 GB free** on drive C:
  - Python environment: about 6 GB, in `C:\Users\<you>\fedecs_env`;
  - datasets and outputs: in `C:\Users\<you>\fedecs_local`.
- Internet: the datasets are downloaded from huggingface.co.
- RAM: 8 GB works for 10-client jobs. Jobs with 50 or 100 clients want 16 GB or more.

## 1. Get the code (once; again when it changes)
On GitHub (github.com/Raunak4518/Fedlearning, signed in), click **Code → Download ZIP**. Extract it to a short path,
for example `C:\Fedlearning`.
- A short path avoids Windows' path-length limit.
- If git is installed, `git clone https://github.com/Raunak4518/Fedlearning.git C:\Fedlearning` works too;
  `git pull` then updates it.

## 2. Set up (once per PC, 15-30 minutes)
Open `C:\Fedlearning\fedecs_lt\local` and **double-click `setup_windows.bat`**. It:
1. checks the NVIDIA driver;
2. installs **Python 3.12.10 for your user only**, if no Python 3.12 is found. This needs no administrator rights;
   Flower's simulation needs exactly 3.12 on Windows.
3. creates the environment `C:\Users\<you>\fedecs_env` with fixed versions:
   - PyTorch 2.14.1 with CUDA 12.6;
   - Flower 1.42.0, Ray 2.55.1;
   - NumPy, SciPy, scikit-learn, Matplotlib, pandas, pyarrow, Pillow, ReportLab, UMAP, TensorBoard.
   The full list is in `requirements.txt`.
4. checks that PyTorch sees the GPU.

If Windows asks whether to run the file, choose **More info → Run anyway**. Run it again after a new download of the
code if `requirements.txt` changed. It skips what is already installed.

## 3. Measure what fits on this GPU (once per PC, 30-90 minutes)
**Double-click `run_check.bat`.** It makes one short run (two rounds per stage, all the data) of every kind of run in
the jobs:
- generator methods at 10, 50 and 100 clients;
- Balanced Softmax and similar;
- the HtFLlib methods;
- the grouped baselines.

For each kind it prints **FITS** or **DOES NOT FIT** and writes the result to
`C:\Users\<you>\fedecs_local\gpu_check.json`.

## 4. Choose a job and run it
**Double-click `list_jobs.bat`.** It shows every job not yet filed: its runs, its Kaggle hours, what is done locally
and whether it fits on this GPU.

**Tell the others which job you take**, and never take one that is running on Kaggle. Then open a Command Prompt in
the folder: in the address bar of `C:\Fedlearning\fedecs_lt\local`, type `cmd` and press Enter. Run:
```
run_job.bat JOB_NAME
run_job.bat JOB_NAME --hours 10      (starts no new run after 10 hours)
```
- **Keep the window open.** To stop, press Ctrl-C: the run in progress is lost, finished runs are kept. The same
  command resumes.
- Keep the laptop plugged in and turn off sleep: Settings → System → Power → Screen and sleep → Never.
- A job whose runs do not fit on this GPU is not started. Give it to a PC with a larger GPU, or to Kaggle.

## 5. Send the results back
The files are in `C:\Users\<you>\fedecs_local\out\JOB_NAME\`.
- Send `JOB_NAME__summary.zip` and `JOB_NAME__report.pdf`, as for Kaggle.
- Keep `__results.zip` and `__models.zip` on Google Drive.

They are filed with `python fedecs_lt/fetch_kaggle.py PATH\JOB_NAME__summary.zip`, which marks them as a local run with
the GPU's name.

## If something goes wrong
| Message | Fix |
|---|---|
| `nvidia-smi not found`, `PyTorch sees no GPU` | install or update the NVIDIA driver (560+) and restart |
| `driver ... too old` | update the NVIDIA driver |
| a missing `.dll` when PyTorch starts | install "Microsoft Visual C++ Redistributable" (x64) from microsoft.com (this one may need an administrator) |
| `CUDA out of memory` in a job | the job is too large for this GPU: give it to a larger GPU or to Kaggle |
| a run dies with `ActorDiedError` or no message | out of RAM: close other programs (browser) and run it again |
| Ray errors at start (`raylet`, `GCS`, timeouts) | close other Python windows, restart the PC, run again; if it repeats, send the text |
| `no connection to huggingface.co` | internet; the datasets are downloaded at run time |
| Windows Defender slows everything | add `C:\Users\<you>\fedecs_local` and `C:\fedecs_ray` to its exclusions (optional) |

## Linux (if a machine has it)
`bash fedecs_lt/local/setup.sh`, then `source ~/fedecs_env/bin/activate` and
`python fedecs_lt/local/run_local.py check | list | JOB_NAME`. On Linux the one-shared-model jobs run too (setup.sh
also installs `requirements_longtaillib.txt`).
