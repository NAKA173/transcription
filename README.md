# Audio to MIDI Transcription

音声ファイル（MP3/WAV）をAIが解析してMIDIファイルに変換するWebアプリケーション。

Spotifyの[Basic Pitch](https://github.com/spotify/basic-pitch)をAI変換エンジンとして使用。

## セットアップ

```bash
pip install -e ".[dev]"
```

## 起動

```bash
uvicorn transcription.main:app --reload
```

ブラウザで http://localhost:8000 にアクセス。

## テスト

```bash
pytest
```

## Docker

```bash
docker build -t transcription .
docker run -p 8000:8000 transcription
```
