@echo off
rem The jobs not yet filed, and whether they fit on this GPU. See README.md.
set PYTHONUTF8=1
"%USERPROFILE%\fedecs_env\Scripts\python.exe" "%~dp0run_local.py" list %*
pause
