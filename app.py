from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import subprocess
import logging
import time
from pydub import AudioSegment
import assemblyai as aai
from deep_translator import GoogleTranslator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- AssemblyAI ----------
ASSEMBLYAI_API_KEY = "c6a6962de12d435a98fe83602b53bbd5"
aai.settings.api_key = ASSEMBLYAI_API_KEY
logging.info("✅ AssemblyAI 已配置 - AI English Teacher V1.2")

# ---------- 翻译器 ----------
def translate_text(text):
    if not text or not text.strip():
        return ""
    for i in range(3):
        try:
            translator = GoogleTranslator(source="en", target="zh-CN")
            result = translator.translate(text)
            if result:
                return result
        except Exception as e:
            logging.warning(f"翻译第{i+1}次失败: {e}")
            time.sleep(1)
    return "【翻译失败】"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 创建目录
os.makedirs("uploads", exist_ok=True)
os.makedirs("chunks", exist_ok=True)
os.makedirs("videos", exist_ok=True)

app.mount("/chunks", StaticFiles(directory="chunks"), name="chunks")

# ---------- 根路由：返回前端页面 ----------
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# ---------- 音频路由 ----------
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join("chunks", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    return JSONResponse({"error": "文件不存在"}, status_code=404)

# ---------- 视频路由 ----------
@app.get("/video/{filename}")
async def get_video(filename: str):
    file_path = os.path.join("videos", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4")
    return JSONResponse({"error": "视频不存在"}, status_code=404)

# ---------- 提取音频 ----------
def extract_audio(video_path, audio_path):
    cmd = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y']
    subprocess.run(cmd, check=True, capture_output=True)

# ---------- 核心处理 ----------
def process_audio_file(audio_path, host, file_id):
    try:
        config = aai.TranscriptionConfig(speaker_labels=True, language_code="en")
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path, config=config)
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"AssemblyAI 失败: {transcript.error}")
        logging.info(f"✅ 转录完成，共 {len(transcript.utterances)} 句")
    except Exception as e:
        logging.error(f"AssemblyAI 调用失败: {e}")
        raise e

    audio = AudioSegment.from_mp3(audio_path)
    sentences_data = []
    last_end_ms = 0

    for idx, utterance in enumerate(transcript.utterances):
        start_ms = int(utterance.start)
        end_ms = int(utterance.end)
        speaker = utterance.speaker
        text = utterance.text.strip()

        if start_ms < last_end_ms:
            start_ms = last_end_ms + 20
        if end_ms <= start_ms:
            end_ms = start_ms + 500

        chunk_audio = audio[start_ms:end_ms]
        chunk_filename = f"{file_id}_speaker_{speaker}_chunk_{idx+1}.mp3"
        chunk_path = f"chunks/{chunk_filename}"
        chunk_audio.export(chunk_path, format="mp3")

        translation = translate_text(text)

        sentences_data.append({
            "index": idx + 1,
            "speaker": speaker,
            "text": text,
            "translation": translation,
            "audio_url": f"{host}/audio/{chunk_filename}",
            "start": start_ms,
            "end": end_ms
        })
        last_end_ms = end_ms

    return {
        "status": "success",
        "full_text": transcript.text,
        "speaker_count": len(set(s["speaker"] for s in sentences_data)),
        "sentences": sentences_data
    }

# ---------- 音频上传 ----------
@app.post("/upload")
async def upload_audio(request: Request, file: UploadFile = File(...)):
    host = f"{request.url.scheme}://{request.url.hostname}:{request.url.port}"
    file_id = str(uuid.uuid4())
    original_path = f"uploads/{file_id}_{file.filename}"
    with open(original_path, "wb") as buffer:
        buffer.write(await file.read())
    logging.info(f"📤 音频已保存")

    try:
        result = process_audio_file(original_path, host, file_id)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

# ---------- 视频上传 ----------
@app.post("/upload_video")
async def upload_video(request: Request, file: UploadFile = File(...)):
    host = f"{request.url.scheme}://{request.url.hostname}:{request.url.port}"
    file_id = str(uuid.uuid4())
    video_filename = f"{file_id}_{file.filename}"
    video_path = f"videos/{video_filename}"
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())
    logging.info(f"📹 视频已保存: {video_path}")

    audio_filename = f"{file_id}_temp_audio.mp3"
    audio_path = f"uploads/{audio_filename}"
    try:
        extract_audio(video_path, audio_path)
        logging.info(f"🎵 音频提取成功: {audio_path}")
    except Exception as e:
        logging.error(f"音频提取失败: {e}")
        return JSONResponse({"status": "error", "message": f"音频提取失败: {str(e)}"})

    try:
        result = process_audio_file(audio_path, host, file_id)
        result["video_url"] = f"{host}/video/{video_filename}"
        result["video_filename"] = video_filename
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})