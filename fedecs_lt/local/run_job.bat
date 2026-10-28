@echo off
rem run_job.bat JOB_NAME [--hours H]: runs one job; run it again to resume. See README.md.
if "%~1"=="" (
  echo Usage: run_job.bat JOB_NAME   ^(list_jobs.bat shows the names^)
  exit /b 1
)
set PYTHONUTF8=1
"%USERPROFILE%\fedecs_env\Scripts\python.exe" "%~dp0run_local.py" %*
