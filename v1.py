"""
guessDgroove - Guess the Song from Intro
Streamlit multiplayer app.
Run with: streamlit run v1.py
"""

import os, re, json, uuid, time, shutil, hashlib, random, string, threading
import asyncio, logging, difflib
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import librosa
from pydub import AudioSegment
import yt_dlp
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from dotenv import load_dotenv

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime,
    ForeignKey, Text, JSON as SA_JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./guessdgroove.db")
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
MAX_PLAYERS = int(os.getenv("MAX_PLAYERS", "4"))
CLIP_DURATION_MS = int(os.getenv("CLIP_DURATION_MS", "10000"))
ROUND_TIMEOUT_SEC = int(os.getenv("ROUND_TIMEOUT_SEC", "15"))
POINTS_CORRECT = int(os.getenv("POINTS_CORRECT", "100"))
POINTS_SPEED_BONUS_MAX = int(os.getenv("POINTS_SPEED_BONUS_MAX", "50"))
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("guessdgroove")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base = declarative_base()


class GameRoom(Base):
    __tablename__ = "game_rooms"
    id = Column(String, primary_key=True)
    host_id = Column(String, nullable=False)
    state = Column(String, default="waiting")
    playlist_url = Column(Text)
    playlist_source = Column(String)
    current_round = Column(Integer, default=0)
    total_rounds = Column(Integer, default=0)
    round_started_at = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Player(Base):
    __tablename__ = "players"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    room_id = Column(String, ForeignKey("game_rooms.id"))
    score = Column(Integer, default=0)
    is_host = Column(Integer, default=0)
    has_guessed = Column(Integer, default=0)
    guess = Column(Text, default="")
    guess_time = Column(Float, default=0.0)


class Track(Base):
    __tablename__ = "tracks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    room_id = Column(String, ForeignKey("game_rooms.id"))
    round_number = Column(Integer)
    title = Column(String)
    artist = Column(String)
    album = Column(String)
    source = Column(String)
    clip_path = Column(String)
    meta_data = Column(SA_JSON)


Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Spotify Manager
# ---------------------------------------------------------------------------

class SpotifyManager:
    def __init__(self):
        self.sp = None
        if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
            try:
                self.sp = spotipy.Spotify(
                    auth_manager=SpotifyClientCredentials(
                        client_id=SPOTIFY_CLIENT_ID,
                        client_secret=SPOTIFY_CLIENT_SECRET))
            except Exception as e:
                logger.warning(f"Spotify init failed: {e}")

    def is_spotify_url(self, url: str) -> bool:
        return bool(re.search(r"open\.spotify\.com/playlist|spotify:playlist:", url))

    def extract_playlist_id(self, url: str) -> Optional[str]:
        m = re.search(r"open\.spotify\.com/playlist/([a-zA-Z0-9]+)", url)
        return m.group(1) if m else (
            re.search(r"spotify:playlist:([a-zA-Z0-9]+)", url).group(1) if re.search(r"spotify:playlist:([a-zA-Z0-9]+)", url) else None
        )

    def get_playlist_tracks(self, url: str) -> list[dict]:
        if not self.sp:
            raise ValueError("Spotify credentials not configured in .env")
        pid = self.extract_playlist_id(url)
        if not pid:
            raise ValueError("Invalid Spotify URL")
        results = self.sp.playlist_items(pid, fields="items.track(name,artists,album(name,images),preview_url,id),next")
        tracks = []
        while results:
            for item in results.get("items", []):
                track = item.get("track")
                if not track or not track.get("name"):
                    continue
                artists = track.get("artists", [])
                album = track.get("album", {})
                tracks.append({
                    "title": track["name"],
                    "artist": artists[0]["name"] if artists else "Unknown",
                    "album": album.get("name", ""),
                    "preview_url": track.get("preview_url"),
                    "track_id": track.get("id"),
                    "image_url": (album.get("images") or [{}])[0].get("url"),
                    "source": "spotify",
                })
            results = self.sp.next(results) if results.get("next") else None
        return tracks

    def get_preview_url_from_embed(self, track_id: str) -> Optional[str]:
        try:
            r = requests.get(f"https://open.spotify.com/embed/track/{track_id}",
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code == 200:
                m = re.search(r'"audioPreview"\s*:\s*\{\s*"url"\s*:\s*"([^"]+)"', r.text)
                if m:
                    return m.group(1)
        except Exception as e:
            logger.warning(f"Embed scrape failed: {e}")
        return None

    def download_preview(self, url: str) -> bytes:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content


# ---------------------------------------------------------------------------
# YouTube Manager
# ---------------------------------------------------------------------------

class YouTubeManager:
    def is_youtube_url(self, url: str) -> bool:
        return bool(re.search(r"youtube\.com|youtu\.be", url))

    def is_playlist(self, url: str) -> bool:
        return "list=" in url or "/playlist" in url

    def get_playlist_entries(self, url: str) -> list[dict]:
        opts = {"extract_flat": True, "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            entries = info.get("entries") or [info]
            return [{
                "title": e.get("title", "Unknown"),
                "artist": e.get("uploader", e.get("channel", "Unknown")),
                "id": e.get("id"),
                "url": e.get("webpage_url") or f"https://www.youtube.com/watch?v={e.get('id', '')}",
                "duration": e.get("duration"),
                "source": "youtube",
            } for e in entries if e]

    def download_audio(self, url: str, out_dir: str) -> dict:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
            "quiet": True, "no_warnings": True, "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            fp = os.path.join(out_dir, f"{info['id']}.mp3")
            return {"filepath": fp, "title": info.get("title", ""), "artist": info.get("uploader", info.get("channel", "")), "id": info["id"]}

    def search_and_download(self, query: str, out_dir: str) -> Optional[dict]:
        opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
            "quiet": True, "no_warnings": True, "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=True)
            if not info or not info.get("entries"):
                return None
            e = info["entries"][0]
            fp = os.path.join(out_dir, f"{e['id']}.mp3")
            return {"filepath": fp, "title": e.get("title", ""), "artist": e.get("uploader", e.get("channel", "")), "id": e["id"]}


# ---------------------------------------------------------------------------
# Audio Processor
# ---------------------------------------------------------------------------

class AudioProcessor:
    def find_vocal_onset(self, audio_bytes: bytes) -> float:
        try:
            y, sr = librosa.load(BytesIO(audio_bytes), sr=22050, mono=True)
        except Exception as e:
            logger.warning(f"librosa failed: {e}")
            return 0.0
        if len(y) < sr:
            return 0.0
        try:
            D = librosa.stft(y)
            Dh, _ = librosa.decompose.hpss(D, margin=3.0)
            yh = librosa.istft(Dh, length=len(y))
        except Exception:
            yh = y
        rms = librosa.feature.rms(y=yh, frame_length=2048, hop_length=512)[0]
        if len(rms) == 0:
            return 0.0
        ws = min(50, max(1, len(rms) // 3))
        smooth = np.convolve(rms, np.ones(ws) / ws, mode="same")
        if np.max(smooth) < 1e-6:
            return 0.0
        thresh = 0.3 * np.max(smooth)
        cnt, frame = 0, len(smooth)
        for i, v in enumerate(smooth):
            if v >= thresh:
                cnt += 1
                if cnt >= 10:
                    frame = i - 9
                    break
            else:
                cnt = 0
        onset = float(librosa.frames_to_time(frame, sr=sr, hop_length=512))
        try:
            oe = librosa.onset.onset_strength(y=yh, sr=sr, hop_length=512)
            of = librosa.onset.onset_detect(onset_envelope=oe, sr=sr, wait=30, pre_max=30, post_max=30, delta=0.3)
            ot = librosa.frames_to_time(of, sr=sr)
            for t in ot:
                if len(ot[(ot >= t) & (ot < t + 2.0)]) >= 3:
                    onset = min(onset, float(t))
                    break
        except Exception:
            pass
        return max(0.0, onset)

    def create_clip(self, audio_bytes: bytes, start_sec: float, end_sec: float) -> bytes:
        audio = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
        clip = audio[int(start_sec * 1000):int(end_sec * 1000)]
        if len(clip) < CLIP_DURATION_MS:
            clip += AudioSegment.silent(duration=CLIP_DURATION_MS - len(clip))
        elif len(clip) > CLIP_DURATION_MS:
            clip = clip[:CLIP_DURATION_MS]
        clip = clip.fade_in(50).fade_out(500)
        buf = BytesIO()
        clip.export(buf, format="mp3", bitrate="128k")
        return buf.getvalue()

    def process(self, audio_bytes: bytes) -> tuple[float, float, float, bytes]:
        dur = 0.0
        try:
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
            dur = len(audio) / 1000.0
        except Exception:
            pass
        onset = self.find_vocal_onset(audio_bytes)
        if dur <= 10.0:
            s, e = 0.0, dur
        elif onset >= 10.0:
            s, e = onset - 10.0, onset
        else:
            s, e = 0.0, 10.0
        clip = self.create_clip(audio_bytes, s, e)
        return onset, s, e, clip


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\b(feat|ft|featuring|remix|original|radio.?edit|explicit|clean|version)\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


def is_correct(guess: str, title: str, artist: str) -> bool:
    ng, nt, na = normalize(guess), normalize(title), normalize(artist)
    if not ng or not nt:
        return False
    if ng == nt or nt in ng or ng in nt:
        return True
    if difflib.SequenceMatcher(None, ng, nt).ratio() >= 0.8:
        return True
    return f"{na} {nt}" in ng


def calc_points(ok: bool, elapsed_ms: int) -> int:
    if not ok:
        return 0
    sf = max(0.0, 1.0 - (elapsed_ms / (ROUND_TIMEOUT_SEC * 1000)))
    return POINTS_CORRECT + int(POINTS_SPEED_BONUS_MAX * sf)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def room_code() -> str:
    chars = (string.ascii_uppercase + string.digits).replace("O", "").replace("I", "").replace("0", "").replace("1", "")
    return "".join(random.choices(chars, k=4))


def get_db():
    return Session()


def init_session():
    for k in ["pid", "name", "code", "host", "stage", "page"]:
        if k not in st.session_state:
            st.session_state[k] = "" if k in ("pid", "name", "code") else (False if k == "host" else "home")


# ---------------------------------------------------------------------------
# Managers (singletons)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_spotify():
    return SpotifyManager()


@st.cache_resource
def get_youtube():
    return YouTubeManager()


@st.cache_resource
def get_audio():
    return AudioProcessor()


# ---------------------------------------------------------------------------
# Page renders
# ---------------------------------------------------------------------------

def page_home():
    st.markdown("""
    <div style='text-align:center;padding:48px 0 24px'>
        <div style='font-size:64px;margin-bottom:8px'>🎵</div>
        <h1 style='font-size:48px;font-weight:800;background:linear-gradient(135deg,#8b5cf6,#c084fc,#f472b6);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0'>
            guessDgroove
        </h1>
        <p style='color:#9090b0;font-size:18px;margin-top:8px'>Guess the song from its intro</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎮  Create Game", use_container_width=True, type="primary"):
            st.session_state.page = "create"
            st.rerun()
    with col2:
        if st.button("🔗  Join Game", use_container_width=True):
            st.session_state.page = "join"
            st.rerun()


def page_create():
    st.markdown("<h2 style='text-align:center;margin-bottom:24px'>Create Game</h2>",
                unsafe_allow_html=True)
    name = st.text_input("Your name", max_chars=20, key="create_name")
    if st.button("Create Room", type="primary", use_container_width=True):
        if not name.strip():
            st.error("Enter your name")
            return
        pid = str(uuid.uuid4())
        code = room_code()
        db = get_db()
        try:
            db.add(GameRoom(id=code, host_id=pid, state="waiting"))
            db.add(Player(id=pid, name=name.strip(), room_id=code, is_host=1))
            db.commit()
        finally:
            db.close()
        st.session_state.pid = pid
        st.session_state.name = name.strip()
        st.session_state.code = code
        st.session_state.host = True
        st.session_state.page = "lobby"
        st.rerun()
    if st.button("← Back"):
        st.session_state.page = "home"
        st.rerun()


def page_join():
    st.markdown("<h2 style='text-align:center;margin-bottom:24px'>Join Game</h2>",
                unsafe_allow_html=True)
    name = st.text_input("Your name", max_chars=20, key="join_name")
    code = st.text_input("Room code (e.g. ABCD)", max_chars=4,
                         key="join_code").strip().upper()

    if st.button("Join Room", type="primary", use_container_width=True):
        if not name.strip() or len(code) != 4:
            st.error("Enter your name and a 4-character room code")
            return
        db = get_db()
        try:
            room = db.query(GameRoom).filter(GameRoom.id == code).first()
            if not room:
                st.error("Room not found")
                return
            if room.state != "waiting":
                st.error("Game already in progress")
                return
            pcount = db.query(Player).filter(Player.room_id == code).count()
            if pcount >= MAX_PLAYERS:
                st.error("Room is full")
                return
            pid = str(uuid.uuid4())
            db.add(Player(id=pid, name=name.strip(), room_id=code, is_host=0))
            db.commit()
        finally:
            db.close()
        st.session_state.pid = pid
        st.session_state.name = name.strip()
        st.session_state.code = code
        st.session_state.host = False
        st.session_state.page = "lobby"
        st.rerun()
    if st.button("← Back"):
        st.session_state.page = "home"
        st.rerun()


def refresh_players():
    db = get_db()
    try:
        ps = db.query(Player).filter(Player.room_id == st.session_state.code).order_by(Player.is_host.desc()).all()
        room = db.query(GameRoom).filter(GameRoom.id == st.session_state.code).first()
        return ps, room
    finally:
        db.close()


def page_lobby():
    ps, room = refresh_players()
    if not room:
        st.error("Room closed")
        st.session_state.page = "home"
        st.rerun()
        return

    st.markdown(f"""
    <div style='text-align:center;padding:16px 0'>
        <p style='color:#606080;font-size:12px;text-transform:uppercase;letter-spacing:2px'>Room Code</p>
        <div style='font-size:48px;font-weight:800;letter-spacing:12px;color:#8b5cf6;cursor:pointer'
             onclick='navigator.clipboard.writeText("{room.id}")' title='Click to copy'>{room.id}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Players ({len(ps)}/{MAX_PLAYERS})**")
    for p in ps:
        col1, col2 = st.columns([4, 1])
        with col1:
            label = f"👤 {p.name}"
            if p.is_host:
                label += " 🏠"
            if p.id == st.session_state.pid:
                label += " (you)"
            st.markdown(f"<div style='padding:6px 12px;background:#1a1a2e;border-radius:8px;margin:2px 0'>{label}</div>",
                        unsafe_allow_html=True)
        with col2:
            if p.is_host:
                st.markdown("<span style='color:#8b5cf6;font-size:12px;font-weight:700'>HOST</span>",
                            unsafe_allow_html=True)

    if st.session_state.host and room.state == "waiting":
        st.divider()
        url = st.text_input("Paste YouTube/Spotify playlist URL", key="playlist_url")
        if st.button("📂  Load Playlist", use_container_width=True) and url.strip():
            with st.spinner("Fetching playlist..."):
                try:
                    spotify = get_spotify()
                    youtube = get_youtube()
                    if spotify.is_spotify_url(url):
                        tracks = spotify.get_playlist_tracks(url)
                        source = "spotify"
                    elif youtube.is_youtube_url(url):
                        if youtube.is_playlist(url):
                            tracks = youtube.get_playlist_entries(url)
                        else:
                            info = youtube.download_audio(url, str(CACHE_DIR))
                            tracks = [{"title": info["title"], "artist": info["artist"],
                                       "source": "youtube", "url": url}]
                        source = "youtube"
                    else:
                        st.error("Unsupported URL")
                        return

                    db = get_db()
                    try:
                        room.playlist_url = url
                        room.playlist_source = source
                        room.total_rounds = len(tracks)
                        room.state = "lobby"
                        for i, t in enumerate(tracks):
                            db.add(Track(room_id=room.id, round_number=i + 1, title=t.get("title", ""),
                                         artist=t.get("artist", ""), album=t.get("album", ""),
                                         source=t.get("source", source), meta_data=t))
                        db.commit()
                    finally:
                        db.close()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    tracks = []
    db = get_db()
    try:
        tracks = db.query(Track).filter(Track.room_id == st.session_state.code).order_by(Track.round_number).all()
    finally:
        db.close()

    if tracks:
        st.markdown(f"**Playlist ({len(tracks)} songs)**")
        for t in tracks[:20]:
            st.markdown(f"<div style='padding:4px 12px;background:#12122a;border-radius:6px;margin:2px 0'>"
                        f"<span style='color:#606080'>{t.round_number}.</span> "
                        f"<strong>{t.title}</strong> "
                        f"<span style='color:#9090b0'>{t.artist}</span></div>",
                        unsafe_allow_html=True)
        if len(tracks) > 20:
            st.caption(f"+ {len(tracks) - 20} more")

        if st.session_state.host and room.state == "lobby" and st.button("▶️  Start Game", type="primary", use_container_width=True):
            db = get_db()
            try:
                room = db.query(GameRoom).filter(GameRoom.id == st.session_state.code).first()
                room.state = "loading"
                room.current_round = 0
                db.commit()
            finally:
                db.close()
            threading.Thread(target=process_clips, args=(st.session_state.code,), daemon=True).start()
            st.session_state.page = "playing"
            st.rerun()

    if st.button("🚪 Leave"):
        db = get_db()
        try:
            db.query(Player).filter(Player.id == st.session_state.pid).delete()
            left = db.query(Player).filter(Player.room_id == st.session_state.code).count()
            if left == 0:
                db.query(Track).filter(Track.room_id == st.session_state.code).delete()
                db.query(GameRoom).filter(GameRoom.id == st.session_state.code).delete()
                CacheManager.clear_room_cache(st.session_state.code)
        finally:
            db.close()
        for k in ["pid", "name", "code", "host", "page"]:
            st.session_state[k] = "" if k in ("pid", "name", "code") else (False if k == "host" else "home")
        st.rerun()

    # Auto-refresh for non-hosts waiting
    if not st.session_state.host and room.state == "lobby":
        time.sleep(2)
        st.rerun()


class CacheManager:
    @staticmethod
    def save_clip(room_code: str, rn: int, data: bytes) -> Path:
        d = CACHE_DIR / room_code
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"r{rn:03d}.mp3"
        p.write_bytes(data)
        return p

    @staticmethod
    def clip_path(room_code: str, rn: int) -> Optional[Path]:
        p = CACHE_DIR / room_code / f"r{rn:03d}.mp3"
        return p if p.exists() else None

    @staticmethod
    def clear_room_cache(room_code: str):
        d = CACHE_DIR / room_code
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def clear_all():
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Game processing thread
# ---------------------------------------------------------------------------

def process_clips(code: str):
    """Run in a thread to process all clips for a room."""
    db = get_db()
    try:
        tracks = db.query(Track).filter(Track.room_id == code).order_by(Track.round_number).all()
        room = db.query(GameRoom).filter(GameRoom.id == code).first()
        if not room:
            return
        temp = str(CACHE_DIR / code / "tmp")
        os.makedirs(temp, exist_ok=True)

        spotify = SpotifyManager()
        youtube = YouTubeManager()
        audio = AudioProcessor()

        for t in tracks:
            try:
                ab = None
                meta = t.meta_data or {}
                if meta.get("source") == "spotify":
                    pu = meta.get("preview_url")
                    if not pu and meta.get("track_id"):
                        pu = spotify.get_preview_url_from_embed(meta["track_id"])
                    if pu:
                        ab = spotify.download_preview(pu)
                    else:
                        q = f"{meta.get('artist', '')} - {meta.get('title', '')}"
                        r = youtube.search_and_download(q, temp)
                        if r and os.path.exists(r["filepath"]):
                            ab = Path(r["filepath"]).read_bytes()
                elif meta.get("source") == "youtube":
                    u = meta.get("url", "")
                    r = youtube.download_audio(u, temp)
                    if r and os.path.exists(r["filepath"]):
                        ab = Path(r["filepath"]).read_bytes()

                if not ab:
                    logger.warning(f"No audio for {t.title}")
                    continue

                _, _, _, clip = audio.process(ab)
                CacheManager.save_clip(code, t.round_number, clip)
                t.clip_path = str(CacheManager.clip_path(code, t.round_number))
                db.commit()
            except Exception as e:
                logger.error(f"Clip error {t.title}: {e}")

        shutil.rmtree(temp, ignore_errors=True)
        room.state = "playing"
        room.current_round = 0
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Playing page
# ---------------------------------------------------------------------------

def page_playing():
    init_session()
    db = get_db()
    try:
        room = db.query(GameRoom).filter(GameRoom.id == st.session_state.code).first()
        ps = db.query(Player).filter(Player.room_id == st.session_state.code).all()
        tracks = db.query(Track).filter(Track.room_id == st.session_state.code).order_by(Track.round_number).all()
    finally:
        db.close()

    if not room or not tracks:
        st.warning("Game room not found")
        st.session_state.page = "home"
        st.rerun()
        return

    # Check if clips are processed
    ready_count = sum(1 for t in tracks if t.clip_path and Path(t.clip_path).exists())
    total = len(tracks)
    clips_ready = ready_count == total and total > 0

    # Loading state — clips being processed in background
    if room.state == "loading":
        st.markdown("### ⏳ Processing audio clips...")
        st.progress(ready_count / total if total > 0 else 0)
        st.caption(f"{ready_count} / {total} clips ready")
        time.sleep(2)
        st.rerun()
        return

    # Clips done but round not started yet — start round 1
    if clips_ready and room.current_round == 0:
        db = get_db()
        try:
            room = db.query(GameRoom).filter(GameRoom.id == st.session_state.code).first()
            room.current_round = 1
            room.round_started_at = time.time()
            db.commit()
        finally:
            db.close()
        st.rerun()
        return

    # Not ready and not loading — wait
    if not clips_ready:
        st.info("Waiting for host to start the game...")
        time.sleep(2)
        st.rerun()
        return

    # Game logic
    cr = room.current_round

    if cr > room.total_rounds:
        st.session_state.page = "scoreboard"
        db = get_db()
        try:
            room = db.query(GameRoom).filter(GameRoom.id == st.session_state.code).first()
            room.state = "ended"
            db.commit()
        finally:
            db.close()
        st.rerun()
        return

    track = next((t for t in tracks if t.round_number == cr), None)
    if not track:
        return

    my_player = next((p for p in ps if p.id == st.session_state.pid), None)
    if not my_player:
        st.error("Player not found")
        st.session_state.page = "home"
        st.rerun()
        return

    all_guessed = all(p.has_guessed for p in ps)
    time_up = time.time() - room.round_started_at > ROUND_TIMEOUT_SEC

    # Show scores sidebar
    sorted_ps = sorted(ps, key=lambda p: p.score, reverse=True)

    # Layout
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown(f"**Round {cr}/{room.total_rounds}**")
        st.divider()
        for p in sorted_ps:
            label = f"{'👑' if p.is_host else '👤'} {p.name}"
            if p.id == st.session_state.pid:
                label += " (you)"
            extra = f" ✅" if p.has_guessed else (" ⏳" if p.id != st.session_state.pid else "")
            st.markdown(f"<div style='padding:6px 10px;background:#1a1a2e;border-radius:8px;margin:2px 0;"
                        f"{'border:1px solid #8b5cf6' if p.id == st.session_state.pid else ''}'>"
                        f"<div style='font-size:14px'>{label}</div>"
                        f"<div style='font-size:20px;font-weight:700;color:#8b5cf6'>{p.score}</div>"
                        f"{'<div style=color:#22c55e;font-size:12px>✅ Guessed</div>' if p.has_guessed else ''}</div>",
                        unsafe_allow_html=True)

    with col2:
        if cr != st.session_state.get("displayed_round"):
            st.session_state.displayed_round = cr

        # Timer
        elapsed = time.time() - room.round_started_at
        remaining = max(0, ROUND_TIMEOUT_SEC - elapsed)
        pct = (remaining / ROUND_TIMEOUT_SEC) * 100
        color = "#22c55e" if remaining > 10 else "#f59e0b" if remaining > 5 else "#ef4444"
        st.markdown(f"""
        <div style='margin-bottom:24px'>
            <div style='height:6px;background:#12122a;border-radius:3px;overflow:hidden'>
                <div style='height:100%;width:{pct}%;background:{color};border-radius:3px;transition:width 1s linear'></div>
            </div>
            <div style='text-align:right;font-size:14px;font-weight:700;color:{color}'>{int(remaining)}s</div>
        </div>
        """, unsafe_allow_html=True)

        # Audio player
        clip_path = CacheManager.clip_path(st.session_state.code, cr)
        if clip_path:
            with open(clip_path, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)

        st.markdown("<p style='text-align:center;font-size:18px;font-weight:600'>🎧 What song is this?</p>",
                    unsafe_allow_html=True)

        # Check if round result should be shown
        show_result = all_guessed or time_up or st.session_state.get("round_over", False)

        if not show_result:
            guess = st.text_input("Your guess", key="guess_input", disabled=my_player.has_guessed,
                                  placeholder="Type song name...")
            if st.button("Submit Guess", type="primary", disabled=my_player.has_guessed, use_container_width=True):
                if guess.strip():
                    db = get_db()
                    try:
                        p = db.query(Player).filter(Player.id == st.session_state.pid).first()
                        if p and not p.has_guessed:
                            p.has_guessed = 1
                            p.guess = guess.strip()
                            p.guess_time = time.time()
                            db.commit()
                    finally:
                        db.close()
                    st.rerun()
        else:
            st.session_state.round_over = False

            # Show round result
            db = get_db()
            try:
                ps_updated = db.query(Player).filter(Player.room_id == st.session_state.code).all()
            finally:
                db.close()

            correct_ids = []
            result_items = []
            for p in ps_updated:
                ok = is_correct(p.guess, track.title, track.artist)
                elapsed_ms = int((p.guess_time - room.round_started_at) * 1000) if p.guess_time else ROUND_TIMEOUT_SEC * 1000
                pts = calc_points(ok, elapsed_ms)
                if ok:
                    correct_ids.append(p.id)
                    score = POINTS_CORRECT + pts
                else:
                    score = 0
                result_items.append((p, ok, pts))

                # Update score in DB
                db2 = get_db()
                try:
                    pp = db2.query(Player).filter(Player.id == p.id).first()
                    if pp and ok:
                        pp.score += score
                    pp.has_guessed = 0
                    pp.guess = ""
                    pp.guess_time = 0.0
                    db2.commit()
                finally:
                    db2.close()

            st.markdown(f"""
            <div style='text-align:center;padding:16px;background:#1a1a2e;border-radius:12px;margin-bottom:16px'>
                <div style='font-size:48px'>{'🎉' if st.session_state.pid in correct_ids else '😢'}</div>
                <h3 style='margin:8px 0'>{track.title}</h3>
                <p style='color:#9090b0'>{track.artist}</p>
            </div>
            """, unsafe_allow_html=True)

            for p, ok, pts in result_items:
                col_a, col_b, col_c = st.columns([2, 3, 1])
                with col_a:
                    st.markdown(f"**{p.name}**{' (you)' if p.id == st.session_state.pid else ''}")
                with col_b:
                    st.markdown(f"<span style='color:#606080'>{p.guess or '—'}</span>", unsafe_allow_html=True)
                with col_c:
                    st.markdown(f"<span style='color:{'#22c55e' if ok else '#ef4444'};font-weight:700'>{f'+{pts}' if ok else '+0'}</span>",
                                unsafe_allow_html=True)

            # Next / Final round button
            is_last = cr >= room.total_rounds
            btn_label = "🏁  See Final Results" if is_last else "Next Round ➡️"
            if st.button(btn_label, type="primary", use_container_width=True):
                db = get_db()
                try:
                    room = db.query(GameRoom).filter(GameRoom.id == st.session_state.code).first()
                    if room:
                        room.current_round += 1
                        room.round_started_at = time.time()
                        db.commit()
                finally:
                    db.close()
                st.rerun()

        # Auto-refresh for non-guessed players
        if not show_result and not my_player.has_guessed:
            time.sleep(1)
            st.rerun()


def page_scoreboard():
    db = get_db()
    try:
        ps = db.query(Player).filter(Player.room_id == st.session_state.code).order_by(Player.score.desc()).all()
    finally:
        db.close()

    if not ps:
        st.warning("No scores available")
        st.session_state.page = "home"
        st.rerun()
        return

    winner = ps[0]
    medals = {0: "🥇", 1: "🥈", 2: "🥉"}

    st.markdown(f"""
    <div style='text-align:center;padding:32px 0'>
        <div style='font-size:64px'>👑</div>
        <h1 style='font-size:36px;font-weight:800;background:linear-gradient(135deg,#fbbf24,#f59e0b);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:8px 0'>{winner.name}</h1>
        <p style='font-size:24px;font-weight:700;color:#8b5cf6'>{winner.score} points</p>
    </div>
    """, unsafe_allow_html=True)

    for i, p in enumerate(ps):
        medal = medals.get(i, f"#{i + 1}")
        border = "2px solid #8b5cf6" if p.id == st.session_state.pid else "1px solid #2a2a5a"
        bg = "rgba(251,191,36,0.05)" if i == 0 else "#1a1a2e"
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:12px;padding:14px 18px;background:{bg};
                    border-radius:10px;border:{border};margin:4px 0'>
            <span style='font-size:24px;min-width:40px'>{medal}</span>
            <span style='flex:1;font-weight:600'>{p.name}</span>
            <span style='font-size:20px;font-weight:700;color:#8b5cf6'>{p.score}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔄 Play Again", type="primary", use_container_width=True):
        db = get_db()
        try:
            db.query(Player).filter(Player.room_id == st.session_state.code).delete()
            db.query(Track).filter(Track.room_id == st.session_state.code).delete()
            db.query(GameRoom).filter(GameRoom.id == st.session_state.code).delete()
        finally:
            db.close()
        CacheManager.clear_room_cache(st.session_state.code)
        for k in ["pid", "name", "code", "host", "page", "displayed_round", "round_over"]:
            st.session_state.pop(k, None)
        st.session_state.page = "home"
        st.rerun()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

st.set_page_config(page_title="guessDgroove", page_icon="🎵", layout="wide")

# Apply dark theme
st.markdown("""
<style>
    .stApp { background: #0a0a1a; color: #e8e8f0; }
    .stTextInput > div > div > input { background: #16163a !important; color: #e8e8f0 !important; border: 1px solid #2a2a5a !important; border-radius: 8px !important; }
    .stTextInput > div > div > input:focus { border-color: #8b5cf6 !important; box-shadow: 0 0 0 3px rgba(139,92,246,0.3) !important; }
    .stButton > button { border-radius: 8px !important; font-weight: 600 !important; }
    .stButton > button[kind="primary"] { background: #8b5cf6 !important; color: white !important; border: none !important; }
    .stButton > button[kind="primary"]:hover { background: #7c3aed !important; box-shadow: 0 4px 16px rgba(139,92,246,0.3) !important; }
    .stButton > button:not([kind]) { background: #16163a !important; color: #e8e8f0 !important; border: 1px solid #2a2a5a !important; }
    .stButton > button:not([kind]):hover { background: #1a1a3e !important; border-color: #606080 !important; }
    div[data-testid="stInfo"] { background: rgba(139,92,246,0.1) !important; border: 1px solid rgba(139,92,246,0.3) !important; color: #e8e8f0 !important; }
    div[data-testid="stAlert"] { background: rgba(239,68,68,0.1) !important; border: 1px solid rgba(239,68,68,0.3) !important; }
    h1, h2, h3, h4 { color: #e8e8f0 !important; }
    .stProgress > div > div > div { background: #8b5cf6 !important; }
    .st-emotion-cache-1c7y2kd { border-color: #2a2a5a !important; }
    .stAudio { display: none; }
</style>
""", unsafe_allow_html=True)

init_session()

# Routing
pages = {
    "home": page_home,
    "create": page_create,
    "join": page_join,
    "lobby": page_lobby,
    "playing": page_playing,
    "scoreboard": page_scoreboard,
}

page_fn = pages.get(st.session_state.page, page_home)
page_fn()
