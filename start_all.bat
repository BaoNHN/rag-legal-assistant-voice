@echo off
REM Starts the voice cloning service (clone-voice-station) and this app together.
REM Voice selection (POST/GET /voice/*) depends on clone-voice-station running on
REM http://127.0.0.1:8090 -- if it's not running, the voice dropdown silently
REM shows no trained/builtin voices (see voice/station_client.py). Trained-voice
REM playback (/voice/speak), when Colab isn't configured/reachable, falls back to
REM voice/rvc_local.py's local conversion -- this now runs in-process inside
REM clone-voice-station's own app.py (cached in RAM there: ~20s cold per speaker,
REM well under 1s once cached), no separate worker process/port needed anymore.

setlocal

set VOICE_STATION_DIR=D:\hoc\project\clone-voice-station

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
