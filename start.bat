@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "exit_code=%errorlevel%"

if not "%exit_code%"=="0" (
  echo.
  echo Startup failed with exit code %exit_code%.
  pause
)

exit /b %exit_code%
