@echo off
setlocal EnableExtensions
title Historical Agent Launcher
cd /d "%~dp0"

set "PYEXE="
set "PYARGS="

rem --- locate a working Python (venv preferred) ---
if exist "venv\Scripts\python.exe" (
    set "PYEXE=venv\Scripts\python.exe"
) else (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PYEXE (
            call :try_run "%%i" ""
            if not errorlevel 1 set "PYEXE=%%i"
        )
    )
    if not defined PYEXE (
        where py >nul 2>nul
        if not errorlevel 1 (
            call :try_run "py" "-3"
            if not errorlevel 1 (
                set "PYEXE=py"
                set "PYARGS=-3"
            )
        )
    )
)

if not defined PYEXE (
    echo.
    echo [Error] Python 3.10+ not found on this machine.
    echo         Install it from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)

echo Using Python: "%PYEXE%" %PYARGS%
echo.
"%PYEXE%" %PYARGS% "%~dp0launcher.py" %*
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
    echo [Finished] The agent exited with code %RC% - see messages above.
)
pause
exit /b %RC%

:try_run
if "%~2"=="" (
    "%~1" -c "pass" >nul 2>nul
) else (
    "%~1" %~2 -c "pass" >nul 2>nul
)
exit /b %errorlevel%
