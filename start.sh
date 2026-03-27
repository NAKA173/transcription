#!/bin/bash
# Audio to MIDI - 起動スクリプト
# 使い方: bash start.sh

set -e

PORT=${1:-8000}

echo "==================================="
echo "  Audio to MIDI 変換サーバー"
echo "==================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 が見つかりません"
    echo "  インストール: https://www.python.org/downloads/"
    exit 1
fi

echo "[1/2] 依存パッケージを確認中..."
pip install --quiet --break-system-packages \
    fastapi uvicorn[standard] python-multipart \
    basic-pitch pretty-midi soundfile 2>/dev/null \
|| pip install --quiet \
    fastapi uvicorn[standard] python-multipart \
    basic-pitch pretty-midi soundfile

echo "[2/2] サーバーを起動中..."
echo ""
echo "-----------------------------------"
echo "  http://localhost:${PORT}"
echo "  ブラウザで上のURLを開いてください"
echo "  停止: Ctrl+C"
echo "-----------------------------------"
echo ""

python3 -m uvicorn transcription.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --http h11 \
    --log-level info
