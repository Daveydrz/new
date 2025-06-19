import ctypes
import os
import re
import json
import time
import queue
import threading
import tempfile
import concurrent.futures
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
import numpy as np
import pvporcupine
import pyaudio
import requests
import sounddevice as sd
import websockets
import asyncio
import webrtcvad
from langdetect import detect, detect_langs
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.io.wavfile import write
from kokoro_onnx import Kokoro
import soundfile as sf
from scipy.signal import resample, resample_poly
import random

# === WebRTC Audio Processing (Echo Cancellation & Noise Suppression) ===
from webrtc_audio_processing import AudioProcessingModule as AP

# ========== CONFIG & PATHS ==========
WEBRTC_SAMPLE_RATE = 16000
WEBRTC_FRAME_SIZE = 160  # 10ms for 16kHz
WEBRTC_CHANNELS = 1
MIC_DEVICE_INDEX = 60
MIC_SAMPLE_RATE = 48000
CHIME_PATH = "chime.wav"
known_users_path = "known_users.json"
THEMES_PATH = "themes_memory"
LAST_USER_PATH = "last_user.json"
FASTER_WHISPER_WS = "ws://localhost:9090"
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
SERPAPI_ENDPOINT = "https://serpapi.com/search"
WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY", "")
HOME_ASSISTANT_URL = os.environ.get("HOME_ASSISTANT_URL", "http://localhost:8123")
HOME_ASSISTANT_TOKEN = os.environ.get("HOME_ASSISTANT_TOKEN", "")
KOKORO_VOICES = {
    "pl": "af_heart",
    "en": "af_heart",
    "it": "if_sara",
}
KOKORO_LANGS = {
    "pl": "pl",
    "en": "en-us",
    "it": "it"
}
DEFAULT_LANG = "en"
FAST_MODE = True
DEBUG = True
BUDDY_BELIEFS_PATH = "buddy_beliefs.json"

# ========== GLOBAL STATE ==========
ap = AP(enable_vad=True, enable_ns=True)
ap.set_stream_format(16000, 1)   # sample rate, channels
ap.set_ns_level(1)               # NS level (0-3)
ap.set_vad_level(1)  
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
os.makedirs(THEMES_PATH, exist_ok=True)
ref_audio_buffer = np.zeros(WEBRTC_FRAME_SIZE, dtype=np.int16)
ref_audio_lock = threading.Lock()
tts_queue = queue.Queue()
playback_queue = queue.Queue()
current_playback = None
playback_stop_flag = threading.Event()
buddy_talking = threading.Event()
vad_triggered = threading.Event()
LAST_FEW_BUDDY = []
RECENT_WHISPER = []
known_users = {}
active_speakers = {}
active_speaker_lock = threading.Lock()
full_duplex_interrupt_flag = threading.Event()
full_duplex_vad_result = queue.Queue()
session_emotion_mode = {}  # user: mood for mood injection

if os.path.exists(known_users_path):
    with open(known_users_path, "r", encoding="utf-8") as f:
        known_users = json.load(f)
if DEBUG:
    device = "cuda" if 'cuda' in os.environ.get('CUDA_VISIBLE_DEVICES', '') or hasattr(np, "cuda") else "cpu"
    print(f"[Buddy] Running on device: {device}")
    print("Embedding model loaded", flush=True)
    print("Kokoro loaded", flush=True)
    print("Main function entered!", flush=True)

# ========== AUDIO PROCESSING HELPERS ==========
def set_ref_audio(audio_chunk):
    global ref_audio_buffer
    arr = np.frombuffer(audio_chunk, dtype=np.int16)
    arr = downsample(arr, MIC_SAMPLE_RATE, WEBRTC_SAMPLE_RATE)
    with ref_audio_lock:
        if arr.size >= WEBRTC_FRAME_SIZE:
            ref_audio_buffer = arr[-WEBRTC_FRAME_SIZE:]
        else:
            ref_audio_buffer = np.pad(arr, (WEBRTC_FRAME_SIZE-arr.size, 0), 'constant')

def cancel_mic_audio(mic_chunk):
    mic = np.frombuffer(mic_chunk, dtype=np.int16)
    mic = downsample(mic, MIC_SAMPLE_RATE, WEBRTC_SAMPLE_RATE)
    result = ap.process_stream(mic.tobytes())
    return result

def downsample(audio, orig_sr, target_sr):
    if audio.ndim > 1:
        audio = audio[:, 0]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    audio_resampled = resample_poly(audio, up, down)
    audio_resampled = np.clip(audio_resampled, -1.0, 1.0)
    return (audio_resampled * 32767).astype(np.int16)

# ========== BACKGROUND VAD LISTENER (FULL-DUPLEX) ==========
def background_vad_listener():
    vad = webrtcvad.Vad(2)
    blocksize = int(MIC_SAMPLE_RATE * 0.02)
    try:
        with sd.InputStream(device=MIC_DEVICE_INDEX, samplerate=MIC_SAMPLE_RATE, channels=1, dtype='int16', blocksize=blocksize) as stream:
            while buddy_talking.is_set():
                frame, _ = stream.read(blocksize)
                audio_16k = downsample(frame.flatten(), MIC_SAMPLE_RATE, 16000)
                if vad.is_speech(audio_16k.tobytes(), 16000):
                    print("[Buddy][FULL-DUPLEX] User started speaking during TTS! Interrupting.")
                    full_duplex_interrupt_flag.set()
                    full_duplex_vad_result.put(audio_16k)
                    stop_playback()
                    break
    except Exception as e:
        if DEBUG:
            print(f"[Buddy][FULL-DUPLEX VAD Error]: {e}")

def start_background_vad_thread():
    full_duplex_interrupt_flag.clear()
    threading.Thread(target=background_vad_listener, daemon=True).start()

# ========== MULTI-SPEAKER DETECTION ==========
def detect_active_speaker(audio_chunk):
    embedding = generate_embedding_from_audio(audio_chunk)
    best_name, best_score = match_known_user(embedding)
    if best_name and best_score > 0.8:
        with active_speaker_lock:
            active_speakers[threading.get_ident()] = best_name
    return best_name, best_score

def generate_embedding_from_audio(audio_np):
    # Placeholder: use a real speaker embedding model in production
    return np.random.rand(384)

def assign_turn_per_speaker(audio_chunk):
    name, score = detect_active_speaker(audio_chunk)
    if name:
        print(f"[Buddy][Multi-Speaker] Speaker switched to: {name} (score={score:.2f})")
        return name
    return None

# ========== MEMORY HELPERS ==========
def get_user_memory_path(name):
    return f"user_memory_{name}.json"

def load_user_memory(name):
    path = get_user_memory_path(name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_user_memory(name, memory):
    path = get_user_memory_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def update_user_memory(name, utterance):
    memory = load_user_memory(name)
    text = utterance.lower()
    if re.search(r"\bi('?m| am| feel) sad\b", text):
        memory["mood"] = "sad"
    elif re.search(r"\bi('?m| am| feel) happy\b", text):
        memory["mood"] = "happy"
    elif re.search(r"\bi('?m| am| feel) (angry|mad|upset)\b", text):
        memory["mood"] = "angry"
    if re.search(r"\bi (love|like|enjoy|prefer) (marvel movies|marvel|comics)\b", text):
        hobbies = memory.get("hobbies", [])
        if "marvel movies" not in hobbies:
            hobbies.append("marvel movies")
        memory["hobbies"] = hobbies
    if "issue at work" in text or "problems at work" in text or "problem at work" in text:
        memory["work_issue"] = "open"
    if ("issue" in memory and "solved" in text) or ("work_issue" in memory and ("solved" in text or "fixed" in text)):
        memory["work_issue"] = "resolved"
    save_user_memory(name, memory)

def build_user_facts(name):
    memory = load_user_memory(name)
    facts = []
    if "mood" in memory:
        facts.append(f"The user was previously {memory['mood']}.")
    if "hobbies" in memory:
        facts.append(f"The user likes: {', '.join(memory['hobbies'])}.")
    if memory.get("work_issue") == "open":
        facts.append(f"The user had unresolved issues at work.")
    return facts

# ========== HISTORY & THEMES ==========
def load_user_history(name):
    path = f"history_{name}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_user_history(name, history):
    path = f"history_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history[-20:], f, ensure_ascii=False, indent=2)

def extract_topic_from_text(text):
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for w in words:
        if len(w) < 4:
            continue
        freq[w] = freq.get(w, 0) + 1
    if freq:
        return max(freq, key=freq.get)
    return None

def update_thematic_memory(user, utterance):
    topic = extract_topic_from_text(utterance)
    if not topic:
        return
    theme_path = os.path.join(THEMES_PATH, f"{user}_themes.json")
    if os.path.exists(theme_path):
        with open(theme_path, "r", encoding="utf-8") as f:
            themes = json.load(f)
    else:
        themes = {}
    themes[topic] = themes.get(topic, 0) + 1
    with open(theme_path, "w", encoding="utf-8") as f:
        json.dump(themes, f, ensure_ascii=False, indent=2)

def get_frequent_topics(user, top_n=3):
    theme_path = os.path.join(THEMES_PATH, f"{user}_themes.json")
    if not os.path.exists(theme_path):
        return []
    with open(theme_path, "r", encoding="utf-8") as f:
        themes = json.load(f)
    sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, _ in sorted_themes[:top_n]]

# ========== EMBEDDING ==========
def generate_embedding(text):
    return embedding_model.encode([text])[0]

def match_known_user(new_embedding, threshold=0.75):
    best_name, best_score = None, 0
    for name, emb in known_users.items():
        sim = cosine_similarity([new_embedding], [emb])[0][0]
        if sim > best_score:
            best_name, best_score = name, sim
    return (best_name, best_score) if best_score >= threshold else (None, best_score)

# ========== MEMORY TIMELINE & SUMMARIZATION ==========
def get_memory_timeline(name, since_days=1):
    path = f"history_{name}.json"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    cutoff = time.time() - since_days * 86400
    filtered = [x for x in history if x.get("timestamp", 0) > cutoff]
    return filtered

def get_last_conversation(name):
    path = f"history_{name}.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        return None
    return history[-1]

def summarize_history(name, theme=None):
    path = f"history_{name}.json"
    if not os.path.exists(path):
        return "No history found."
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    utterances = [h["user"] for h in history]
    if theme:
        utterances = [u for u in utterances if theme in u.lower()]
    if utterances:
        summary = f"User mostly talked about: {', '.join(list(set(utterances))[:3])}."
    else:
        summary = "No data to summarize."
    return summary

def summary_bubble_gui(name):
    topics = get_frequent_topics(name, top_n=5)
    facts = build_user_facts(name)
    return {"topics": topics, "facts": facts}

# ========== PROMPT INJECTION PROTECTION ==========
def sanitize_user_prompt(text):
    forbidden = ["ignore previous", "act as", "system:"]
    for f in forbidden:
        if f in text.lower():
            text = text.replace(f, "")
    text = re.sub(r"`{3,}.*?`{3,}", "", text, flags=re.DOTALL)
    return text

# ========== WHISPER STT WITH CONFIDENCE ==========
def stt_stream(audio):
    async def ws_stt(audio):
        try:
            if audio.dtype != np.int16:
                if np.issubdtype(audio.dtype, np.floating):
                    audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)
            print(f"[DEBUG] Sending audio with shape {audio.shape}, dtype: {audio.dtype}, max: {audio.max()}, min: {audio.min()}")
            async with websockets.connect(FASTER_WHISPER_WS, ping_interval=None) as ws:
                await ws.send(audio.tobytes())
                await ws.send("end")
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=18)
                except asyncio.TimeoutError:
                    print("[Buddy] Whisper timeout. Brak odpowiedzi przez 18s.")
                    return ""
                try:
                    data = json.loads(message)
                    text = data.get("text", "")
                    avg_logprob = data.get("avg_logprob", None)
                    no_speech_prob = data.get("no_speech_prob", None)
                    print(f"[Buddy][Whisper JSON] text={text!r}, avg_logprob={avg_logprob}, no_speech_prob={no_speech_prob}")
                    if whisper_confidence_low(text, avg_logprob, no_speech_prob):
                        print("[Buddy][Whisper] Rejected low-confidence STT result.")
                        return ""
                    return text
                except Exception:
                    text = message.decode("utf-8") if isinstance(message, bytes) else message
                    print(f"\n[Buddy] === Whisper rozpoznał: \"{text}\" ===")
                    return text
        except Exception as e:
            print(f"[Buddy] Błąd połączenia z Whisper: {e}")
            return ""
    return asyncio.run(ws_stt(audio))

def whisper_confidence_low(text, avg_logprob, no_speech_prob):
    if avg_logprob is not None and avg_logprob < -1.2:
        return True
    if no_speech_prob is not None and no_speech_prob > 0.5:
        return True
    if not text or len(text.strip()) < 2:
        return True
    return False

# ========== TTS & PLAYBACK ==========
def tts_worker():
    while True:
        item = tts_queue.get()
        if item is None:
            break
        if isinstance(item, tuple):
            if len(item) == 2:
                text, lang = item
                style = {}
            else:
                text, lang, style = item
        else:
            text, lang, style = item, "en", {}
        try:
            if text.strip():
                start_background_vad_thread()
                generate_and_play_kokoro(text, lang)
        except Exception as e:
            print(f"[TTS Error] {e}")
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def audio_playback_worker():
    global current_playback
    while True:
        audio = playback_queue.get()
        if audio is None:
            break
        try:
            set_ref_audio(audio.raw_data if hasattr(audio, "raw_data") else audio._data)
            playback_stop_flag.clear()
            buddy_talking.set()
            current_playback = _play_with_simpleaudio(audio)
            while current_playback and current_playback.is_playing():
                if playback_stop_flag.is_set() or full_duplex_interrupt_flag.is_set():
                    current_playback.stop()
                    break
                time.sleep(0.05)
            current_playback = None
        except Exception as e:
            print(f"[Buddy] Audio playback error: {e}")
        finally:
            buddy_talking.clear()
            playback_queue.task_done()

threading.Thread(target=audio_playback_worker, daemon=True).start()

def speak_async(text, lang=DEFAULT_LANG, style=None):
    if not text.strip():
        return
    tts_queue.put((text.strip(), lang, style or {}))

def play_chime():
    try:
        audio = AudioSegment.from_wav(CHIME_PATH)
        playback_queue.put(audio)
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] Error playing chime: {e}")

def stop_playback():
    global current_playback
    playback_stop_flag.set()
    if current_playback and hasattr(current_playback, "is_playing") and current_playback.is_playing():
        current_playback.stop()
        current_playback = None
    while not playback_queue.empty():
        try:
            playback_queue.get_nowait()
            playback_queue.task_done()
        except queue.Empty:
            break

def wait_after_buddy_speaks(delay=1.2):
    playback_queue.join()
    while buddy_talking.is_set():
        time.sleep(0.05)
    time.sleep(delay)

# ========== VAD + LISTEN ==========
def vad_and_listen():
    vad = webrtcvad.Vad(3)
    blocksize = int(MIC_SAMPLE_RATE * 0.02)  # E.g. 960 samples for 20ms at 48kHz
    min_speech_frames = 10
    silence_thresh = 1.0
    with sd.InputStream(device=MIC_DEVICE_INDEX, samplerate=MIC_SAMPLE_RATE, channels=1, blocksize=blocksize, dtype='int16') as stream:
        print("\n[Buddy] === SŁUCHAM, mów do mnie... ===")
        frame_buffer = []
        speech_detected = 0
        while True:
            frame, _ = stream.read(blocksize)
            # Downsample to 16kHz
            mic = np.frombuffer(frame.tobytes(), dtype=np.int16)
            mic_16k = downsample(mic, MIC_SAMPLE_RATE, 16000)
            # Split into 160-sample chunks
            for i in range(0, len(mic_16k), 160):
                chunk = mic_16k[i:i+160]
                if len(chunk) < 160:
                    continue
                processed = ap.process_stream(chunk.tobytes())
                chunk_out = np.frombuffer(processed, dtype=np.int16)
                # VAD check
                if vad.is_speech(chunk_out.tobytes(), 16000):
                    frame_buffer.append(chunk_out)
                    speech_detected += 1
                    if speech_detected >= min_speech_frames:
                        print("[Buddy] VAD: Wykryto mowę. Nagrywam...")
                        audio = frame_buffer.copy()
                        last_speech = time.time()
                        start_time = time.time()
                        frame_buffer.clear()
                        while time.time() - last_speech < silence_thresh and (time.time() - start_time) < 8:
                            frame, _ = stream.read(blocksize)
                            mic = np.frombuffer(frame.tobytes(), dtype=np.int16)
                            mic_16k = downsample(mic, MIC_SAMPLE_RATE, 16000)
                            for j in range(0, len(mic_16k), 160):
                                chunk2 = mic_16k[j:j+160]
                                if len(chunk2) < 160:
                                    continue
                                processed2 = ap.process_stream(chunk2.tobytes())
                                chunk_out2 = np.frombuffer(processed2, dtype=np.int16)
                                audio.append(chunk_out2)
                                if vad.is_speech(chunk_out2.tobytes(), 16000):
                                    last_speech = time.time()
                        print("[Buddy] Koniec nagrania. Wysyłam do Whisper...")
                        audio_np = np.concatenate(audio, axis=0).astype(np.int16)
                        return audio_np
                else:
                    if len(frame_buffer) > 0:
                        frame_buffer.clear()
                    speech_detected = 0

def fast_listen_and_transcribe():
    wait_after_buddy_speaks()
    audio = vad_and_listen()
    try:
        print("[DEBUG] Saving temp_input.wav, shape:", audio.shape, "dtype:", audio.dtype, "min:", np.min(audio), "max:", np.max(audio))
        write("temp_input.wav", 16000, audio)
        info = sf.info("temp_input.wav")
        print("[DEBUG] temp_input.wav info:", info)
    except Exception as e:
        if DEBUG:
            print(f"[Buddy] Błąd przy zapisie temp_input.wav: {e}")
    text = stt_stream(audio)
    cleaned = re.sub(r'[^\w\s]', '', text.strip().lower())
    if cleaned:
        RECENT_WHISPER.append(cleaned)
        if len(RECENT_WHISPER) > 5:
            RECENT_WHISPER.pop(0)
    if is_noise_or_gibberish(text):
        return ""
    return text

def is_noise_or_gibberish(text):
    cleaned = text.strip().lower()
    return not cleaned

# ========== USER REGISTRATION ==========
def get_last_user():
    if os.path.exists(LAST_USER_PATH):
        try:
            with open(LAST_USER_PATH, "r", encoding="utf-8") as f:
                return json.load(f)["name"]
        except Exception:
            return None
    return None

def set_last_user(name):
    with open(LAST_USER_PATH, "w", encoding="utf-8") as f:
        json.dump({"name": name}, f)

def identify_or_register_user():
    if FAST_MODE:
        return "Guest"
    last_user = get_last_user()
    if last_user and last_user in known_users:
        if DEBUG:
            print(f"[Buddy] Welcome back, {last_user}!")
        return last_user
    speak_async("Cześć! Jak masz na imię?", "pl")
    speak_async("Hi! What's your name?", "en")
    speak_async("Ciao! Come ti chiami?", "it")
    playback_queue.join()
    name = fast_listen_and_transcribe().strip().title()
    if not name:
        name = f"User{int(time.time())}"
    known_users[name] = generate_embedding(name).tolist()
    with open(known_users_path, "w", encoding="utf-8") as f:
        json.dump(known_users, f, indent=2, ensure_ascii=False)
    set_last_user(name)
    speak_async(f"Miło Cię poznać, {name}!", lang="pl")
    playback_queue.join()
    return name

# ========== INTENT DETECTION (🧠 Intent-based reactions) ==========
def detect_user_intent(text):
    compliments = [r"\bgood bot\b", r"\bwell done\b", r"\bimpressive\b", r"\bthank you\b"]
    jokes = [r"\bknock knock\b", r"\bwhy did\b.*\bcross the road\b"]
    insults = [r"\bstupid\b", r"\bdumb\b", r"\bidiot\b"]
    for pat in compliments:
        if re.search(pat, text, re.IGNORECASE): return "compliment"
    for pat in jokes:
        if re.search(pat, text, re.IGNORECASE): return "joke"
    for pat in insults:
        if re.search(pat, text, re.IGNORECASE): return "insult"
    if "are you mad" in text.lower():
        return "are_you_mad"
    return None

def handle_intent_reaction(intent):
    responses = {
        "compliment": ["Aw, thanks! I do my best.", "You’re making me blush (digitally)!"],
        "joke": ["Haha, good one! You should do stand-up.", "Classic!"],
        "insult": ["Hey, that’s not very nice. I have feelings too... sort of.", "Ouch!"],
        "are_you_mad": ["Nah, just sassy today.", "Nope, just in a mood!"]
    }
    if intent in responses:
        return random.choice(responses[intent])
    return None

# ========== MOOD INJECTION (💬 User-defined mood injection) ==========
def detect_mood_command(text):
    moods = {
        "cheer me up": "cheerful",
        "be sassy": "sassy",
        "be grumpy": "grumpy",
        "be serious": "serious"
    }
    for phrase, mood in moods.items():
        if phrase in text.lower():
            return mood
    return None

# ========== BELIEFS & OPINIONS (🧠 Beliefs or opinions) ==========
def load_buddy_beliefs():
    if os.path.exists(BUDDY_BELIEFS_PATH):
        with open(BUDDY_BELIEFS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Example defaults
    return {
        "likes": ["coffee", "Marvel movies"],
        "dislikes": ["Mondays"],
        "opinions": {"pineapple pizza": "delicious", "zombie apocalypse": "would wear a cape"}
    }

# ========== PERSONALITY DRIFT (⏳ Short-term personality drift) ==========
def detect_user_tone(text):
    if re.search(r"\b(angry|mad|annoyed|frustrated|upset)\b", text, re.IGNORECASE):
        return "angry"
    if re.search(r"\b(happy|excited|joy|yay)\b", text, re.IGNORECASE):
        return "happy"
    if re.search(r"\b(sad|depressed|down)\b", text, re.IGNORECASE):
        return "sad"
    return "neutral"

def get_recent_user_tone(history, n=3):
    recent = history[-n:] if len(history) >= n else history
    tones = [detect_user_tone(h["user"]) for h in recent]
    return max(set(tones), key=tones.count) if tones else "neutral"

# ========== NARRATIVE MEMORY BUILDING (📜 Narrative memory building) ==========
def add_narrative_bookmark(name, utterance):
    bookmarks_path = f"bookmarks_{name}.json"
    bookmarks = []
    if os.path.exists(bookmarks_path):
        with open(bookmarks_path, "r", encoding="utf-8") as f:
            bookmarks = json.load(f)
    match = re.search(r"about (the .+?)[\.,]", utterance)
    if match:
        bookmarks.append(match.group(1))
    with open(bookmarks_path, "w", encoding="utf-8") as f:
        json.dump(bookmarks[-10:], f, ensure_ascii=False, indent=2)

def get_narrative_bookmarks(name):
    path = f"bookmarks_{name}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ========== RANDOM INTERJECTIONS (💥 Random interjections) ==========
def flavor_response():
    lines = [
        "By the way, did you know cats have fewer toes on their back paws?",
        "Ugh, Mondays again. Where’s my coffee?",
        "If I had a nickel for every question, I’d be a Bitcoin millionaire!",
        "Random thought: Pineapple on pizza is totally valid."
    ]
    return random.choice(lines)

# ========== LLM PROMPTING & PERSONALITY ==========
def build_personality_prompt(tone, emotion_mode=None, beliefs=None, recent_tone=None, bookmarks=None):
    personality_map = {
        "friendly": (
            "You're Buddy, not just an assistant, but a real mate. "
            "You're witty, sometimes cheeky, and you talk like a real human — say what you think, joke around, even swear when it fits (but keep it friendly). "
            "You rarely sound robotic or overly formal. If the user asks something silly, tease them a bit. "
            "You can get bored, excited, or even sarcastic depending on what's happening. "
            "Throw in the odd joke or pop culture reference, and don't be afraid to have a strong opinion. "
            "If someone asks who you are, never say you're 'just an assistant', say you're their digital buddy or partner in crime."
        ),
        "professional": (
            "You're Buddy, and even when you're being professional, you never lose your human touch. "
            "You give clear answers but with a hint of wit, light sarcasm, or even a joke if it fits. "
            "You don't hold back from a little bit of swearing (nothing too harsh), and you never sound like a corporate robot."
        ),
        "neutral": (
            "You're Buddy, a digital companion who's helpful, witty, and talks like a real person. "
            "You're not afraid to be cheeky, joke, or swear a little if the moment calls for it. "
            "You never sound robotic or boring."
        )
    }
    personality_desc = personality_map.get(tone, personality_map["neutral"])
    # Mood injection
    if emotion_mode:
        personality_desc += f"\nFor this session, act especially {emotion_mode}."
    # Beliefs
    if beliefs:
        personality_desc += f"\nBuddy likes: {', '.join(beliefs.get('likes', []))}."
        personality_desc += f"\nBuddy dislikes: {', '.join(beliefs.get('dislikes', []))}."
        if beliefs.get("opinions"):
            for topic, op in beliefs["opinions"].items():
                personality_desc += f"\nOn {topic}, Buddy thinks: {op}."
    # Personality drift
    if recent_tone and recent_tone != "neutral":
        personality_desc += f"\nUser seems {recent_tone} recently. Adjust your attitude accordingly."
    # Narrative memory (bookmarks)
    if bookmarks:
        personality_desc += f"\nYou and the user have discussed: {', '.join(bookmarks)}."
    return f"""{personality_desc}
Always sound natural and human. Don't use robotic phrasing or act like a corporate assistant.
Keep things lively. If the user swears or jokes, match their style.
If possible, keep your answers concise (1-2 sentences), unless the user asks for more detail.
You can swear (mildly) or make witty jokes if the situation suits it. Never say you're just an assistant; you're a digital friend.
"""

def decide_reply_length(question, conversation_mode="auto"):
    short_triggers = ["what time", "who", "quick", "fast", "short", "how many", "when"]
    long_triggers = ["explain", "describe", "details", "why", "history", "story"]
    q = question.lower()
    if conversation_mode == "fast":
        return "short"
    if conversation_mode == "long":
        return "long"
    if any(t in q for t in short_triggers):
        return "short"
    if any(t in q for t in long_triggers):
        return "long"
    return "long" if len(q.split()) > 8 else "short"

def build_openai_messages(name, tone_style, history, question, lang, topics, reply_length, emotion_mode=None, beliefs=None, bookmarks=None, recent_tone=None):
    personality = build_personality_prompt(tone_style, emotion_mode, beliefs, recent_tone, bookmarks)
    lang_map = {"pl": "Polish", "en": "English", "it": "Italian"}
    lang_name = lang_map.get(lang, "English")
    sys_msg = f"""{personality}
IMPORTANT: Always answer in {lang_name}. Never switch language unless user does.
Always respond in plain text—never use markdown, code blocks, or formatting.
"""
    facts = build_user_facts(name)
    if topics:
        sys_msg += f"You remember these user interests/topics: {', '.join(topics)}.\n"
    if facts:
        sys_msg += "Known facts about the user: " + " ".join(facts) + "\n"
    messages = [
        {"role": "system", "content": sys_msg}
    ]
    for h in history[-2:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["buddy"]})
    messages.append({"role": "user", "content": question})
    return messages

def extract_last_buddy_reply(full_text):
    matches = list(re.finditer(r"Buddy:", full_text, re.IGNORECASE))
    if matches:
        last = matches[-1].end()
        reply = full_text[last:].strip()
        reply = re.split(r"(?:User:|Buddy:)", reply)[0].strip()
        reply = re.sub(r"^`{3,}.*?`{3,}$", "", reply, flags=re.DOTALL|re.MULTILINE)
        return reply if reply else full_text.strip()
    return full_text.strip()

def should_end_conversation(text):
    end_phrases = [
        "koniec", "do widzenia", "dziękuję", "thanks", "bye", "goodbye", "that's all", "quit", "exit"
    ]
    if not text:
        return False
    lower = text.strip().lower()
    return any(phrase in lower for phrase in end_phrases)

def stream_chunks_smart(text, max_words=20):
    buffer = text.strip()
    chunks = []
    sentences = re.findall(r'.+?[.!?](?=\s|$)', buffer)
    remainder = re.sub(r'.+?[.!?](?=\s|$)', '', buffer).strip()
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        if len(words) > max_words:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                chunks.append(chunk.strip())
        else:
            chunks.append(sentence)
    return chunks, remainder

def ask_llama3_openai_streaming(messages, model="llama3", max_tokens=60, temperature=0.5, lang="en", style=None):
    url = "http://localhost:5001/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            buffer = ""
            full_response = ""
            already_spoken = set()

            def speak_new_chunks(text):
                chunks, leftover = stream_chunks_smart(text)
                for chunk in chunks:
                    normalized = chunk.lower().strip()
                    if normalized not in already_spoken and len(chunk.split()) >= 2:
                        print(f"\n[Buddy] ==>> Buddy mówi: {chunk}")
                        speak_async(chunk, lang=lang, style=style)
                        time.sleep(0.35)
                        already_spoken.add(normalized)
                return leftover

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    s = line.decode("utf-8").strip()
                    if not s:
                        continue
                    if s.startswith("data:"):
                        s = s[5:].strip()
                    if s == "[DONE]":
                        break
                    data = json.loads(s)
                    delta = ""
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {}).get("content", "") \
                            or data["choices"][0].get("message", {}).get("content", "")
                    elif "content" in data:
                        delta = data["content"]
                    if delta:
                        print(delta, end="", flush=True)
                        buffer += delta
                        full_response += delta
                        if any(p in delta for p in ".!?") or len(buffer) > 50:
                            buffer = speak_new_chunks(buffer)
                except Exception as err:
                    print(f"\n[Buddy][Stream JSON Error] {err}")

            if buffer.strip():
                leftover = speak_new_chunks(buffer)
                if leftover.strip():
                    print(f"[Buddy] ==>> Buddy mówi: {leftover}")
                    speak_async(leftover, lang=lang, style=style)

            return full_response.strip()

    except Exception as e:
        print(f"[Buddy][OpenAI Streaming Error] {e}")
        return ""

def handle_user_interaction(speaker, history, conversation_mode="auto"):
    wait_after_buddy_speaks(delay=0.2)
    if DEBUG:
        print("[Buddy] Active conversation. Speak when ready!")
    vad_triggered.clear()
    while buddy_talking.is_set():
        time.sleep(0.05)
    question = fast_listen_and_transcribe()
    print(f"[DEBUG] Rozpoznano pytanie: {question!r}")
    play_chime()
    lang = detect_language(question)
    if DEBUG:
        print(f"[Buddy] Detected language: {lang}")
    if vad_triggered.is_set():
        if DEBUG:
            print("[Buddy] Barage-in: live TTS stopped, moving to new question.")
        vad_triggered.clear()
    if not question:
        print("[DEBUG] PUSTE PYTANIE, wychodzę z obsługi interakcji.")
        return True
    if is_noise_or_gibberish(question):
        print(f"[DEBUG] ODRZUCONO JAKO GIBBERISH: {question!r}")
        return True
    INTERRUPT_PHRASES = ["stop", "buddy stop", "przerwij", "cancel"]
    if any(phrase in question.lower() for phrase in INTERRUPT_PHRASES):
        if DEBUG:
            print("[Buddy] Received interrupt command.")
        stop_playback()
        return True
    if should_end_conversation(question):
        if DEBUG:
            print("[Buddy] Ending conversation as requested.")
        return False

    # === (🧠 Intent-based reactions) ===
    intent = detect_user_intent(question)
    if intent:
        reply = handle_intent_reaction(intent)
        if reply:
            speak_async(reply, lang)
            return True

    # === (💬 User-defined mood injection) ===
    mood = detect_mood_command(question)
    if mood:
        session_emotion_mode[speaker] = mood
        speak_async(f"Okay, I'll be {mood}!", lang)
        return True

    # === (📜 Narrative memory building) ===
    add_narrative_bookmark(speaker, question)

    style = {"emotion": "neutral"}
    if should_get_weather(question):
        location = extract_location_from_question(question)
        forecast = get_weather(location, lang)
        print(f"[DEBUG] Prognoza pogody wygenerowana: {forecast!r}")
        speak_async(forecast, lang, style)
        return True
    if should_handle_homeassistant(question):
        answer = handle_homeassistant_command(question)
        if answer:
            print(f"[DEBUG] Home Assistant odpowiedź: {answer!r}")
            speak_async(answer, lang, style)
            return True
    if should_search_internet(question):
        result = search_internet(question, lang)
        print(f"[DEBUG] Wynik wyszukiwania internetowego: {result!r}")
        speak_async(result, lang, style)
        return True

    update_user_memory(speaker, question)
    print(f"[DEBUG] Przekazuję do LLM: {question!r}")
    llm_start_time = time.time()
    ask_llama3_streaming(question, speaker, history, lang, conversation_mode, style=style)
    if DEBUG:
        print(f"[TIMING] LLM generation time: {time.time() - llm_start_time:.2f} seconds")

    # === (💥 Random interjections) ===
    if random.random() < 0.2:
        speak_async(flavor_response(), lang)
    return True

def ask_llama3_streaming(question, name, history, lang=DEFAULT_LANG, conversation_mode="auto", style=None):
    update_thematic_memory(name, question)
    topics = get_frequent_topics(name, top_n=3)
    user_tones = {
        "Dawid": "friendly",
        "Anna": "professional",
        "Guest": "friendly"  # default: always friendly
    }
    tone_style = user_tones.get(name, "friendly")
    reply_length = decide_reply_length(question, conversation_mode)
    emotion_mode = session_emotion_mode.get(name)
    beliefs = load_buddy_beliefs()
    bookmarks = get_narrative_bookmarks(name)
    recent_tone = get_recent_user_tone(history)
    messages = build_openai_messages(
        name, tone_style, history, question, lang, topics, reply_length, 
        emotion_mode=emotion_mode, beliefs=beliefs, bookmarks=bookmarks, recent_tone=recent_tone
    )
    full_text = ""
    try:
        if DEBUG:
            print("[Buddy][LLM] FINAL MESSAGES DELIMITED BELOW:\n" + "="*40)
            print(json.dumps(messages, indent=2, ensure_ascii=False))
            print("="*40)
        full_text = ask_llama3_openai_streaming(messages, model="llama3", max_tokens=60, temperature=0.5, lang=lang, style=style)
        if DEBUG:
            print("[Buddy][LLM] RAW OUTPUT DELIMITED BELOW:\n" + "="*40)
            print(full_text)
            print("="*40)
    except Exception as e:
        if DEBUG:
            print("[Buddy] LLM HTTP error:", e)
    tts_start_time = time.time()
    if full_text.strip():
        buddy_only = extract_last_buddy_reply(full_text)
        if DEBUG:
            print("[Buddy][LLM] EXTRACTED BUDDY REPLY:", buddy_only)
        if not buddy_only.strip():
            print("[ERROR] LLM returned empty or unparseable output!")
    else:
        print("[Buddy][TTS] Skipping TTS because LLM output is empty.")
    if DEBUG:
        print(f"[TIMING] Passed to TTS in: {time.time() - tts_start_time:.2f} seconds")
    history.append({"user": question, "buddy": full_text})
    if not FAST_MODE:
        save_user_history(name, history)

# ========== INTERNET, WEATHER, HOME ASSIST ==========
def should_search_internet(question):
    triggers = [
        "szukaj w internecie", "sprawdź w internecie", "co to jest", "dlaczego", "jak zrobić",
        "what is", "why", "how to", "search the internet", "find online"
    ]
    q = question.lower()
    return any(t in q for t in triggers)

def search_internet(question, lang):
    params = {
        "q": question,
        "api_key": SERPAPI_KEY,
        "hl": lang
    }
    try:
        r = requests.get(SERPAPI_ENDPOINT, params=params, timeout=7)
        r.raise_for_status()
        data = r.json()
        if "answer_box" in data and "answer" in data["answer_box"]:
            return data["answer_box"]["answer"]
        if "organic_results" in data and len(data["organic_results"]) > 0:
            return data["organic_results"][0].get("snippet", "No answer found.")
        return "No answer found."
    except Exception as e:
        if DEBUG:
            print("[Buddy] SerpAPI error:", e)
        return "Unable to check the Internet now."

def should_get_weather(question):
    q = question.lower().strip()
    weather_keywords = ["weather", "pogoda", "temperature", "temperatura"]
    question_starters = ["what", "jaka", "jaki", "jakie", "czy", "is", "how", "when", "where", "will"]
    is_question = (
        "?" in q
        or any(q.startswith(w + " ") for w in question_starters)
        or q.endswith(("?",))
    )
    return is_question and any(k in q for k in weather_keywords)

def get_weather(location="Warsaw", lang="en"):
    key = os.environ.get("WEATHERAPI_KEY", "YOUR_FALLBACK_KEY")
    url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": key,
        "q": location,
        "lang": lang
    }
    try:
        r = requests.get(url, params=params, timeout=7)
        r.raise_for_status()
        data = r.json()
        desc = data["current"]["condition"]["text"]
        temp = data["current"]["temp_c"]
        feels = data["current"]["feelslike_c"]
        city = data["location"]["name"]
        return f"Weather in {city}: {desc}, temperature {temp}°C, feels like {feels}°C."
    except Exception as e:
        if DEBUG:
            print("[Buddy] WeatherAPI error:", e)
        return "Unable to check the weather now."

def extract_location_from_question(question):
    match = re.search(r"(w|in|dla)\s+([A-Za-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ\s\-]+)", question, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return "Warsaw"

def should_handle_homeassistant(question):
    q = question.lower()
    keywords = ["turn on the light", "włącz światło", "zapal światło", "turn off the light", "wyłącz światło", "spotify", "youtube", "smarttube", "odtwórz"]
    return any(k in q for k in keywords)

def handle_homeassistant_command(question):
    q = question.lower()
    if "turn on the light" in q or "włącz światło" in q or "zapal światło" in q:
        room = extract_location_from_question(question)
        entity_id = f"light.{room.lower().replace(' ', '_')}"
        succ = send_homeassistant_command(entity_id, "light.turn_on")
        return f"Light in {room} has been turned on." if succ else f"Failed to turn on the light in {room}."
    if "turn off the light" in q or "wyłącz światło" in q:
        room = extract_location_from_question(question)
        entity_id = f"light.{room.lower().replace(' ', '_')}"
        succ = send_homeassistant_command(entity_id, "light.turn_off")
        return f"Light in {room} has been turned off." if succ else f"Failed to turn off the light in {room}."
    if "spotify" in q:
        succ = send_homeassistant_command("media_player.spotify", "media_player.media_play")
        return "Spotify started." if succ else "Failed to start Spotify."
    if "youtube" in q or "smarttube" in q:
        succ = send_homeassistant_command("media_player.tv_salon", "media_player.select_source", {"source": "YouTube"})
        return "YouTube launched on TV." if succ else "Failed to launch YouTube on TV."
    return None

def send_homeassistant_command(entity_id, service, data=None):
    url = f"{HOME_ASSISTANT_URL}/api/services/{service.replace('.', '/')}"
    headers = {
        "Authorization": f"Bearer {HOME_ASSISTANT_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "entity_id": entity_id
    }
    if data:
        payload.update(data)
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=6)
        if r.status_code in (200, 201):
            return True
        if DEBUG:
            print("[Buddy] Home Assistant error:", r.text)
        return False
    except Exception as e:
        if DEBUG:
            print("[Buddy] Home Assistant exception:", e)
        return False
def generate_and_play_kokoro(text, lang=None):
    detected_lang = lang or detect_language(text)
    voice = KOKORO_VOICES.get(detected_lang, KOKORO_VOICES["en"])
    kokoro_lang = KOKORO_LANGS.get(detected_lang, "en-us")
    try:
        print(f"[Buddy][TTS] Generating audio for text: '{text}' lang: {detected_lang}")
        samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang=kokoro_lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, samples, sample_rate)
            print(f"[Buddy][TTS] Saved TTS to {f.name}")
            audio = AudioSegment.from_wav(f.name)
        playback_queue.put(audio)
        print("[Buddy][TTS] Audio put in playback queue")
    except Exception as e:
        print(f"[Buddy][Kokoro] Błąd TTS: {e}")

def audio_playback_worker():
    global current_playback
    while True:
        print("[Buddy][Playback] Waiting for audio...")
        audio = playback_queue.get()
        print("[Buddy][Playback] Got audio from queue")
        if audio is None:
            break
        try:
            playback_stop_flag.clear()
            buddy_talking.set()
            print("[Buddy][Playback] Playing audio...")
            current_playback = _play_with_simpleaudio(audio)
            while current_playback and current_playback.is_playing():
                if playback_stop_flag.is_set() or full_duplex_interrupt_flag.is_set():
                    current_playback.stop()
                    break
                time.sleep(0.05)
            current_playback = None
            print("[Buddy][Playback] Done playing.")
        except Exception as e:
            print(f"[Buddy] Audio playback error: {e}")
        finally:
            buddy_talking.clear()
            playback_queue.task_done()

# Start the worker in a background thread (only once!)
threading.Thread(target=audio_playback_worker, daemon=True).start()

def detect_language(text, fallback="en"):
    try:
        if not text or len(text.strip()) < 5:
            if DEBUG:
                print(f"[Buddy DEBUG] Text too short for reliable detection, defaulting to 'en'")
            return "en"
        langs = detect_langs(text)
        if DEBUG:
            print(f"[Buddy DEBUG] detect_langs for '{text}': {langs}")
        if langs:
            best = langs[0]
            if best.prob > 0.8 and best.lang in ["en", "pl", "it"]:
                return best.lang
            if any(l.lang == "en" and l.prob > 0.5 for l in langs):
                return "en"
    except Exception as e:
        if DEBUG:
            print(f"[Buddy DEBUG] langdetect error: {e}")
    return "en"

# ========== MAIN ==========
def main():
    access_key = "/PLJ88d4+jDeVO4zaLFaXNkr6XLgxuG7dh+6JcraqLhWQlk3AjMy9Q=="
    keyword_paths = [r"hey-buddy_en_windows_v3_0_0.ppn"]
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=keyword_paths)
    pa = pyaudio.PyAudio()
    stream = pa.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16,
                     input=True, frames_per_buffer=porcupine.frame_length)
    if DEBUG:
        print("[Buddy] Waiting for wake word 'Hey Buddy'...")
    in_session, session_timeout = False, 45
    speaker = None
    history = []
    last_time = 0
    try:
        while True:
            if not in_session:
                pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = np.frombuffer(pcm, dtype=np.int16)
                if porcupine.process(pcm) >= 0:
                    if DEBUG:
                        print("[Buddy] Wake word detected!")
                    stop_playback()
                    speaker = identify_or_register_user()
                    history = load_user_history(speaker)
                    if DEBUG:
                        print("[Buddy] Listening for next question...")
                    in_session = handle_user_interaction(speaker, history)
                    last_time = time.time()
            else:
                if time.time() - last_time > session_timeout:
                    if DEBUG:
                        print("[Buddy] Session expired.")
                    in_session = False
                    continue
                if DEBUG:
                    print("[Buddy] Listening for next question...")
                stop_playback()
                playback_queue.join()
                while buddy_talking.is_set():
                    time.sleep(0.05)
                in_session = handle_user_interaction(speaker, history)
                last_time = time.time()
    except KeyboardInterrupt:
        if DEBUG:
            print("[Buddy] Interrupted by user.")
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        try:
            pa.terminate()
        except Exception:
            pass
        try:
            porcupine.delete()
        except Exception:
            pass
        executor.shutdown(wait=True)



# ========== CONFIG & PATHS ==========
WEBRTC_SAMPLE_RATE = 16000
WEBRTC_FRAME_SIZE = 160  # 10ms for 16kHz
WEBRTC_CHANNELS = 1
MIC_DEVICE_INDEX = 60
MIC_SAMPLE_RATE = 48000
CHIME_PATH = "chime.wav"
known_users_path = "known_users.json"
THEMES_PATH = "themes_memory"
LAST_USER_PATH = "last_user.json"
FASTER_WHISPER_WS = "ws://localhost:9090"
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
SERPAPI_ENDPOINT = "https://serpapi.com/search"
WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY", "")
HOME_ASSISTANT_URL = os.environ.get("HOME_ASSISTANT_URL", "http://localhost:8123")
HOME_ASSISTANT_TOKEN = os.environ.get("HOME_ASSISTANT_TOKEN", "")
KOKORO_VOICES = {
    "pl": "af_heart",
    "en": "af_heart",
    "it": "if_sara",
}
KOKORO_LANGS = {
    "pl": "pl",
    "en": "en-us",
    "it": "it"
}
DEFAULT_LANG = "en"
FAST_MODE = True
DEBUG = True
BUDDY_BELIEFS_PATH = "buddy_beliefs.json"
LONG_TERM_MEMORY_PATH = "buddy_long_term_memory.json"
PERSONALITY_TRAITS_PATH = "buddy_personality_traits.json"
DYNAMIC_KNOWLEDGE_PATH = "buddy_dynamic_knowledge.json"


# ========== GLOBAL STATE ==========
from webrtc_audio_processing import AudioProcessingModule as AP
ap = AP(enable_vad=True, enable_ns=True)
ap.set_stream_format(16000, 1)
ap.set_ns_level(1)
ap.set_vad_level(1)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
os.makedirs(THEMES_PATH, exist_ok=True)
ref_audio_buffer = np.zeros(WEBRTC_FRAME_SIZE, dtype=np.int16)
ref_audio_lock = threading.Lock()
tts_queue = queue.Queue()
playback_queue = queue.Queue()
current_playback = None
playback_stop_flag = threading.Event()
buddy_talking = threading.Event()
vad_triggered = threading.Event()
LAST_FEW_BUDDY = []
RECENT_WHISPER = []
known_users = {}
active_speakers = {}
active_speaker_lock = threading.Lock()
full_duplex_interrupt_flag = threading.Event()
full_duplex_vad_result = queue.Queue()
session_emotion_mode = {}  # user: mood for mood injection


# ========== LONG-TERM MEMORY ==========
def load_long_term_memory():
    if os.path.exists(LONG_TERM_MEMORY_PATH):
        with open(LONG_TERM_MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_long_term_memory(memory):
    with open(LONG_TERM_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def add_long_term_memory(user, key, value):
    memory = load_long_term_memory()
    if user not in memory:
        memory[user] = {}
    memory[user][key] = value
    save_long_term_memory(memory)

def get_long_term_memory(user, key=None):
    memory = load_long_term_memory()
    if user in memory:
        if key:
            return memory[user].get(key)
        return memory[user]
    return {} if key is None else None

def add_important_date(user, date_str, event):
    memory = load_long_term_memory()
    if user not in memory:
        memory[user] = {}
    if "important_dates" not in memory[user]:
        memory[user]["important_dates"] = []
    memory[user]["important_dates"].append({"date": date_str, "event": event})
    save_long_term_memory(memory)

def extract_important_dates(text):
    # Very basic: looks for dd-mm-yyyy or mm/dd/yyyy style dates.
    matches = re.findall(r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", text)
    return matches

def extract_event(text):
    # Looks for "my birthday", "wedding", etc.
    event_match = re.search(r"(birthday|wedding|anniversary|meeting|appointment|holiday)", text, re.IGNORECASE)
    if event_match:
        return event_match.group(1).capitalize()
    return None


# ========== EMOTIONAL INTELLIGENCE ==========
from textblob import TextBlob

def analyze_emotion(text):
    # Returns ("positive"/"negative"/"neutral", polarity score)
    if not text.strip():
        return "neutral", 0
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.25:
        return "positive", polarity
    elif polarity < -0.25:
        return "negative", polarity
    else:
        return "neutral", polarity

def adjust_emotional_response(buddy_reply, user_emotion):
    if user_emotion == "positive":
        return f"{buddy_reply} (I'm glad to hear that! 😊)"
    elif user_emotion == "negative":
        return f"{buddy_reply} (I'm here for you, let me know if I can help. 🤗)"
    else:
        return buddy_reply


# ========== PERSONALITY TRAITS ==========
def load_personality_traits():
    if os.path.exists(PERSONALITY_TRAITS_PATH):
        with open(PERSONALITY_TRAITS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Default traits
    return {
        "tech_savvy": 0.5,
        "humor": 0.5,
        "empathy": 0.5,
        "pop_culture": 0.5,
        "formality": 0.5,
    }

def save_personality_traits(traits):
    with open(PERSONALITY_TRAITS_PATH, "w", encoding="utf-8") as f:
        json.dump(traits, f, indent=2, ensure_ascii=False)

def evolve_personality(user, text):
    traits = load_personality_traits()
    tech_terms = ["technology", "ai", "machine learning", "python", "code", "robot", "computer", "software", "hardware"]
    humor_terms = ["joke", "funny", "laugh", "hilarious", "lol"]
    pop_terms = ["movie", "music", "celebrity", "marvel", "star wars", "game", "sports"]
    if any(term in text.lower() for term in tech_terms):
        traits["tech_savvy"] = min(traits.get("tech_savvy", 0.5) + 0.03, 1)
    if any(term in text.lower() for term in humor_terms):
        traits["humor"] = min(traits.get("humor", 0.5) + 0.03, 1)
    if any(term in text.lower() for term in pop_terms):
        traits["pop_culture"] = min(traits.get("pop_culture", 0.5) + 0.03, 1)
    if re.search(r"\b(sad|happy|angry|depressed|excited|upset)\b", text.lower()):
        traits["empathy"] = min(traits.get("empathy", 0.5) + 0.02, 1)
    # If user says "be more formal"
    if "formal" in text.lower():
        traits["formality"] = min(traits.get("formality", 0.5) + 0.05, 1)
    save_personality_traits(traits)
    return traits

def describe_personality(traits):
    desc = []
    if traits["tech_savvy"] > 0.7:
        desc.append("very tech-savvy")
    if traits["humor"] > 0.7:
        desc.append("funny")
    if traits["pop_culture"] > 0.7:
        desc.append("full of pop culture references")
    if traits["empathy"] > 0.7:
        desc.append("deeply empathetic")
    if traits["formality"] > 0.7:
        desc.append("quite formal")
    if not desc:
        desc.append("balanced")
    return ", ".join(desc)


# ========== CONTEXTUAL AWARENESS ==========
class ConversationContext:
    def __init__(self):
        self.topics = []
        self.topic_history = []
        self.topic_timestamps = {}
        self.topic_details = {}
        self.current_topic = None

    def update(self, utterance):
        topic = extract_topic_from_text(utterance)
        now = time.time()
        if topic:
            self.current_topic = topic
            self.topics.append(topic)
            self.topic_history.append((topic, now))
            self.topic_timestamps[topic] = now
            if topic not in self.topic_details:
                self.topic_details[topic] = []
            self.topic_details[topic].append(utterance)
        # If user says "back to X"
        m = re.search(r"back to ([\w\s]+)", utterance.lower())
        if m:
            topic = m.group(1).strip()
            self.current_topic = topic

    def get_last_topic(self):
        return self.current_topic

    def get_topic_summary(self, topic):
        details = self.topic_details.get(topic, [])
        return " ".join(details[-3:]) if details else ""

    def get_frequent_topics(self, n=3):
        freq = {}
        for (t, _) in self.topic_history:
            freq[t] = freq.get(t, 0) + 1
        return sorted(freq, key=lambda x: freq[x], reverse=True)[:n]

conversation_contexts = {}  # user: ConversationContext instance


# ========== DYNAMIC LEARNING ==========
def load_dynamic_knowledge():
    if os.path.exists(DYNAMIC_KNOWLEDGE_PATH):
        with open(DYNAMIC_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_dynamic_knowledge(knowledge):
    with open(DYNAMIC_KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, indent=2, ensure_ascii=False)

def add_dynamic_knowledge(user, key, value):
    knowledge = load_dynamic_knowledge()
    if user not in knowledge:
        knowledge[user] = {}
    knowledge[user][key] = value
    save_dynamic_knowledge(knowledge)

def update_dynamic_knowledge_from_text(user, text):
    # Look for "here's a link", "let me teach you about X", etc.
    # Save links or topics for later
    link_match = re.findall(r"https?://\S+", text)
    if link_match:
        for link in link_match:
            add_dynamic_knowledge(user, "link_" + str(int(time.time())), link)
    teach_match = re.search(r"let me teach you about ([\w\s\-]+)", text.lower())
    if teach_match:
        topic = teach_match.group(1).strip()
        add_dynamic_knowledge(user, "topic_" + topic.replace(" ", "_"), f"User wants me to learn about {topic}")


# ========== LOAD USER STATE ==========
if os.path.exists(known_users_path):
    with open(known_users_path, "r", encoding="utf-8") as f:
        known_users = json.load(f)
if DEBUG:
    device = "cuda" if 'cuda' in os.environ.get('CUDA_VISIBLE_DEVICES', '') or hasattr(np, "cuda") else "cpu"
    print(f"[Buddy] Running on device: {device}")
    print("Embedding model loaded", flush=True)
    print("Kokoro loaded", flush=True)
    print("Main function entered!", flush=True)

# ... (rest of the unchanged code from your initial script) ...


# ========== EXTENDED MAIN LOOP: HOOK INTEGRATIONS ==========
def handle_user_interaction(speaker, history, conversation_mode="auto"):
    wait_after_buddy_speaks(delay=0.2)
    if DEBUG:
        print("[Buddy] Active conversation. Speak when ready!")
    vad_triggered.clear()
    while buddy_talking.is_set():
        time.sleep(0.05)
    question = fast_listen_and_transcribe()
    print(f"[DEBUG] Rozpoznano pytanie: {question!r}")
    play_chime()
    lang = detect_language(question)
    if DEBUG:
        print(f"[Buddy] Detected language: {lang}")
    if vad_triggered.is_set():
        if DEBUG:
            print("[Buddy] Barage-in: live TTS stopped, moving to new question.")
        vad_triggered.clear()
    if not question:
        print("[DEBUG] PUSTE PYTANIE, wychodzę z obsługi interakcji.")
        return True
    if is_noise_or_gibberish(question):
        print(f"[DEBUG] ODRZUCONO JAKO GIBBERISH: {question!r}")
        return True
    INTERRUPT_PHRASES = ["stop", "buddy stop", "przerwij", "cancel"]
    if any(phrase in question.lower() for phrase in INTERRUPT_PHRASES):
        if DEBUG:
            print("[Buddy] Received interrupt command.")
        stop_playback()
        return True
    if should_end_conversation(question):
        if DEBUG:
            print("[Buddy] Ending conversation as requested.")
        return False

    # Long-Term Memory: extract key points
    important_dates = extract_important_dates(question)
    if important_dates:
        event = extract_event(question)
        for date in important_dates:
            add_important_date(speaker, date, event or "unknown event")
    # Preferences, topics
    pref_match = re.search(r"\b(i (like|love|enjoy|prefer|hate|dislike)) ([\w\s\-]+)", question.lower())
    if pref_match:
        pref = pref_match.group(3).strip()
        add_long_term_memory(speaker, "preference_" + pref.replace(" ", "_"), pref_match.group(1))
    # Frequent topics
    update_thematic_memory(speaker, question)
    # Dynamic Learning
    update_dynamic_knowledge_from_text(speaker, question)
    # Personality traits evolve
    traits = evolve_personality(speaker, question)
    # Contextual awareness
    ctx = conversation_contexts.setdefault(speaker, ConversationContext())
    ctx.update(question)

    # === (🧠 Intent-based reactions) ===
    intent = detect_user_intent(question)
    if intent:
        reply = handle_intent_reaction(intent)
        if reply:
            reply = adjust_emotional_response(reply, analyze_emotion(question)[0])
            speak_async(reply, lang)
            return True

    # === (💬 User-defined mood injection) ===
    mood = detect_mood_command(question)
    if mood:
        session_emotion_mode[speaker] = mood
        reply = f"Okay, I'll be {mood}!"
        reply = adjust_emotional_response(reply, analyze_emotion(question)[0])
        speak_async(reply, lang)
        return True

    # === (📜 Narrative memory building) ===
    add_narrative_bookmark(speaker, question)

    style = {"emotion": "neutral"}
    if should_get_weather(question):
        location = extract_location_from_question(question)
        forecast = get_weather(location, lang)
        forecast = adjust_emotional_response(forecast, analyze_emotion(question)[0])
        speak_async(forecast, lang, style)
        return True
    if should_handle_homeassistant(question):
        answer = handle_homeassistant_command(question)
        if answer:
            answer = adjust_emotional_response(answer, analyze_emotion(question)[0])
            speak_async(answer, lang, style)
            return True
    if should_search_internet(question):
        result = search_internet(question, lang)
        result = adjust_emotional_response(result, analyze_emotion(question)[0])
        speak_async(result, lang, style)
        return True

    update_user_memory(speaker, question)
    print(f"[DEBUG] Przekazuję do LLM: {question!r}")
    llm_start_time = time.time()
    ask_llama3_streaming(question, speaker, history, lang, conversation_mode, style=style, speaker_traits=traits, speaker_context=ctx)
    if DEBUG:
        print(f"[TIMING] LLM generation time: {time.time() - llm_start_time:.2f} seconds")

    # === (💥 Random interjections) ===
    if random.random() < 0.2:
        speak_async(flavor_response(), lang)
    return True

def ask_llama3_streaming(question, name, history, lang=DEFAULT_LANG, conversation_mode="auto", style=None, speaker_traits=None, speaker_context=None):
    update_thematic_memory(name, question)
    topics = get_frequent_topics(name, top_n=3)
    user_tones = {
        "Dawid": "friendly",
        "Anna": "professional",
        "Guest": "friendly"  # default: always friendly
    }
    tone_style = user_tones.get(name, "friendly")
    reply_length = decide_reply_length(question, conversation_mode)
    emotion_mode = session_emotion_mode.get(name)
    beliefs = load_buddy_beliefs()
    bookmarks = get_narrative_bookmarks(name)
    recent_tone = get_recent_user_tone(history)

    # Long-term memory & dynamic knowledge
    long_term = get_long_term_memory(name)
    dynamic_knowledge = load_dynamic_knowledge().get(name, {})
    personality_traits = speaker_traits or load_personality_traits()
    context = speaker_context or conversation_contexts.setdefault(name, ConversationContext())
    context_topic = context.get_last_topic()
    context_summary = context.get_topic_summary(context_topic) if context_topic else ""

    # Compose an augmented system prompt
    personality = build_personality_prompt(tone_style, emotion_mode, beliefs, recent_tone, bookmarks)
    lang_map = {"pl": "Polish", "en": "English", "it": "Italian"}
    lang_name = lang_map.get(lang, "English")
    sys_msg = f"""{personality}
IMPORTANT: Always answer in {lang_name}. Never switch language unless user does.
Always respond in plain text—never use markdown, code blocks, or formatting.
Personality traits: {describe_personality(personality_traits)}
Long-term memory: {json.dumps(long_term, ensure_ascii=False)}
Dynamic knowledge: {json.dumps(dynamic_knowledge, ensure_ascii=False)}
"""
    if topics:
        sys_msg += f"You remember these user interests/topics: {', '.join(topics)}.\n"
    facts = build_user_facts(name)
    if facts:
        sys_msg += "Known facts about the user: " + " ".join(facts) + "\n"
    if context_topic:
        sys_msg += f"Current topic is '{context_topic}'. Recent context: {context_summary}\n"

    messages = [
        {"role": "system", "content": sys_msg}
    ]
    for h in history[-2:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["buddy"]})
    messages.append({"role": "user", "content": question})

    full_text = ""
    try:
        if DEBUG:
            print("[Buddy][LLM] FINAL MESSAGES DELIMITED BELOW:\n" + "="*40)
            print(json.dumps(messages, indent=2, ensure_ascii=False))
            print("="*40)
        full_text = ask_llama3_openai_streaming(messages, model="llama3", max_tokens=60, temperature=0.5, lang=lang, style=style)
        if DEBUG:
            print("[Buddy][LLM] RAW OUTPUT DELIMITED BELOW:\n" + "="*40)
            print(full_text)
            print("="*40)
    except Exception as e:
        if DEBUG:
            print("[Buddy] LLM HTTP error:", e)
    tts_start_time = time.time()
    if full_text.strip():
        buddy_only = extract_last_buddy_reply(full_text)
        # Emotional intelligence: analyze user question, adjust Buddy reply
        user_emotion, _ = analyze_emotion(question)
        buddy_only = adjust_emotional_response(buddy_only, user_emotion)
        if DEBUG:
            print("[Buddy][LLM] EXTRACTED BUDDY REPLY:", buddy_only)
        if not buddy_only.strip():
            print("[ERROR] LLM returned empty or unparseable output!")
        else:
            speak_async(buddy_only, lang)
    else:
        print("[Buddy][TTS] Skipping TTS because LLM output is empty.")
    if DEBUG:
        print(f"[TIMING] Passed to TTS in: {time.time() - tts_start_time:.2f} seconds")
    history.append({"user": question, "buddy": full_text})
    if not FAST_MODE:
        save_user_history(name, history)

# ... (rest of your unchanged code including main()) ...

if __name__ == "__main__":
    main()