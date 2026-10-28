@echo off
rem Once per PC (30-90 minutes): measures which kinds of run fit on this GPU. See README.md.
set PYTHONUTF8=1
"%USERPROFILE%\fedecs_env\Scripts\python.exe" "%~dp0run_local.py" check %*
pause
