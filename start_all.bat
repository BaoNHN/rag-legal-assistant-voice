@echo off
REM Starts the local cached RVC worker, the voice cloning service
REM (clone-voice-station), and this app together.
REM Voice selection (POST/GET /voice/*) depends on clone-voice-station running on
REM http://127.0.0.1:8090 -- if it's not running, the voice dropdown silently
REM shows no trained/builtin voices (see voice/station_client.py). Trained-voice
REM playback (/voice/speak) also depends on the local RVC worker on
REM http://127.0.0.1:8091 -- clone-voice-station's rvc_endpoint setting points
REM at it for in-process model caching (~65s cold, ~0.5s cached vs. ~65s every
REM call for the subprocess fallback in voice/rvc_local.py); if this worker
REM isn't running, conversion still works, just without caching.

setlocal

set VOICE_STATION_DIR=D:\hoc\project\clone-voice-station

echo =============================================
echo   Starting Local RVC Worker (port 8091)...
echo =============================================
start "Local RVC Worker" cmd /k "cd /d %VOICE_STATION_DIR%\local_rvc && venv\Scripts\python.exe local_rvc_server.py"

echo Waiting for Local RVC Worker to become healthy...
set RETRIES=0
:WAIT_RVC_LOOP
curl -s -o nul -w "%%{http_code}" http://127.0.0.1:8091/health > "%TEMP%\local_rvc_health.txt" 2>nul
set /p RVC_HEALTH_CODE=<"%TEMP%\local_rvc_health.txt"
if "%RVC_HEALTH_CODE%"=="200" goto RVC_READY
set /a RETRIES+=1
if %RETRIES% GEQ 30 (
    echo.
    echo [WARNING] Local RVC Worker did not become healthy after 30s.
    echo           Trained-voice playback will fall back to the slower,
    echo           uncached subprocess path until it's up.
    echo           Check the "Local RVC Worker" window for errors.
    goto START_STATION
)
timeout /t 1 /nobreak >nul
goto WAIT_RVC_LOOP

:RVC_READY
echo Local RVC Worker is up.

:START_STATION
echo.
echo =============================================
echo   Starting Clone Voice Station (port 8090)...
echo =============================================
start "Clone Voice Station" cmd /k "call D:\anaconda3\condabin\conda.bat activate rag_env && cd /d %VOICE_STATION_DIR% && python app.py"

echo Waiting for Clone Voice Station to become healthy...
set RETRIES=0
:WAIT_LOOP
curl -s -o nul -w "%%{http_code}" http://127.0.0.1:8090/api/health > "%TEMP%\voice_station_health.txt" 2>nul
set /p HEALTH_CODE=<"%TEMP%\voice_station_health.txt"
if "%HEALTH_CODE%"=="200" goto STATION_READY
set /a RETRIES+=1
if %RETRIES% GEQ 60 (
    echo.
    echo [WARNING] Clone Voice Station did not become healthy after 60s.
    echo           Voice selection will show no voices until it's up.
    echo           Check the "Clone Voice Station" window for errors.
    goto START_RAG
)
timeout /t 1 /nobreak >nul
goto WAIT_LOOP

:STATION_READY
echo Clone Voice Station is up.

:START_RAG
echo.
echo =============================================
echo   Starting RAG Legal Assistant (port 8000)...
echo =============================================
call D:\anaconda3\condabin\conda.bat activate rag_env
cd /d %~dp0
python app.py

endlocal
