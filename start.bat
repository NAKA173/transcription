@echo off
REM Audio to MIDI - 起動スクリプト (Windows)
REM 使い方: start.bat をダブルクリック

echo ===================================
echo   Audio to MIDI 変換サーバー
echo ===================================
echo.

echo [1/2] 依存パッケージを確認中...
pip install --quiet fastapi "uvicorn[standard]" python-multipart basic-pitch pretty-midi soundfile scipy

echo [2/2] サーバーを起動中...
echo.
echo -----------------------------------
echo   http://localhost:8000
echo   ブラウザで上のURLを開いてください
echo   停止: Ctrl+C
echo -----------------------------------
echo.

python -m uvicorn transcription.main:app --host 0.0.0.0 --port 8000 --http h11 --log-level info

pause
