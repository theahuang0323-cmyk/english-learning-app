from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import subprocess
import logging
from pydub import AudioSegment
import assemblyai as aai
from deep_translator import GoogleTranslator

# ---------- 日志 ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- AssemblyAI 配置 ----------
ASSEMBLYAI_API_KEY = "c6a6962de12d435a98fe83602b53bbd5"
aai.settings.api_key = ASSEMBLYAI_API_KEY
logging.info("✅ AssemblyAI 已配置 - AI English Teacher V1.2")

# ---------- 翻译器 ----------
# 使用函数内创建翻译器，避免启动时初始化失败导致整个翻译关闭

def translate_text(text):
    """
    英文 -> 中文翻译
    """
    if not text or not text.strip():
        return ""

    import time

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

# ---------- FastAPI 应用 ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 创建必要目录
os.makedirs("uploads", exist_ok=True)
os.makedirs("chunks", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# 静态挂载
app.mount("/chunks", StaticFiles(directory="chunks"), name="chunks")

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

# ---------- 工具函数：从视频提取音频 ----------
def extract_audio(video_path, audio_path):
    cmd = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y']
    subprocess.run(cmd, check=True, capture_output=True)

# ---------- 核心处理函数 ----------
def process_audio_file(audio_path, host, file_id):
    """处理音频文件，返回句子数据"""
    try:
        config = aai.TranscriptionConfig(speaker_labels=True, language_code="en")
        transcriber = aai.Transcriber(config=config)
        logging.info("开始调用 AssemblyAI...")
        transcript = transcriber.transcribe(audio_path)
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"AssemblyAI 失败: {transcript.error}")
        logging.info(f"✅ 转录完成，共 {len(transcript.utterances)} 句")
    except Exception as e:
        logging.error(f"AssemblyAI 调用失败: {e}")
        raise e

    audio = AudioSegment.from_mp3(audio_path)
    sentences_data = []
    last_end_ms = 0

    utterances = transcript.utterances or []

    # 兼容没有 speaker 分离结果的情况
    if not utterances and transcript.text:
        class SimpleUtterance:
            start = 0
            end = len(audio)
            speaker = "A"
            text = transcript.text
        utterances = [SimpleUtterance()]

    for idx, utterance in enumerate(utterances):
        start_ms = int(utterance.start)
        end_ms = int(utterance.end)
        speaker = utterance.speaker
        text = utterance.text.strip()

        # 修正重叠
        if start_ms < last_end_ms:
            start_ms = last_end_ms + 20
        if end_ms <= start_ms:
            end_ms = start_ms + 500

        # 切割音频片段
        chunk_audio = audio[start_ms:end_ms]
        chunk_filename = f"{file_id}_speaker_{speaker}_chunk_{idx+1}.mp3"
        chunk_path = f"chunks/{chunk_filename}"
        chunk_audio.export(chunk_path, format="mp3")

        # 翻译
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
        "full_text": transcript.text or "",
        "speaker_count": len(set(s["speaker"] for s in sentences_data)),
        "sentences": sentences_data
    }

# ---------- 音频上传接口 ----------
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
        # 可选删除原始音频
        # os.remove(original_path)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

# ---------- 视频上传接口 ----------
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
        # 添加视频信息
        result["video_url"] = f"{host}/video/{video_filename}"
        result["video_filename"] = video_filename
        # 删除临时音频
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

# ---------- 根路径 ----------
@app.get("/")
def read_root():
    return {"message": "英语学习网站后端已启动（支持音频和视频），请访问 index.html"}