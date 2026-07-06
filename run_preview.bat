@echo off
rem Second instance for UI preview/testing — port 8010, logs to console.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8010
