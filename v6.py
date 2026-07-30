"""
guessDgroove v6 - Song Prelude Extractor with 403 Error Workaround
Extract individual songs from a YouTube video, identify song names from chapters/description,
and play a short intro prelude of each.
Run with: streamlit run v6.py
"""

import os, re, io
from pathlib import Path

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pydub import AudioSegment
from pydub.silence import detect_silence
import yt_dlp

CACHE_DIR = Path("./cache/v6")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _build_opts() -> dict:
    client_order = os.getenv("YT_CLIENT", "android,web,ios").split(",")
    opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": str(CACHE_DIR / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "quiet": True, "no_warnings": True,
        "noplaylist": True,
        "http_headers": {"User-Agent": USER_AGENT},
        "extractor_args": {"youtube": {"player_client": client_order}},
        "retries": 15,
        "fragment_retries": 15,
        "extractor_retries": 5,
        "file_access_retries": 5,
        "ignoreerrors": True,
    }
    cookie_file = os.getenv("YT_COOKIES_FILE", "")
    if cookie_file:
        cookie_path = Path(cookie_file)
        if cookie_path.exists():
            opts["cookiefile"] = str(cookie_path)
    return opts


def download_audio(url: str) -> tuple:
    opts = _build_opts()
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError("YouTube returned no data (video may be private, age-restricted, or geo-blocked)")
        filepath = str(CACHE_DIR / f"{info['id']}.mp3")
        title = info.get("title", "Unknown")
        chapters = info.get("chapters") or []
        description = info.get("description") or ""
        return filepath, title, chapters, description


def ms_to_str(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _merge_boundaries(bounds_ms: list[int], total_ms: int, min_segment_len: int) -> list[int]:
    bounds_ms = sorted(set(b for b in bounds_ms if 0 < b < total_ms))
    merged = [0]
    for b in bounds_ms:
        if b - merged[-1] >= min_segment_len:
            merged.append(b)
    if total_ms - merged[-1] >= min_segment_len:
        merged.append(total_ms)
    elif len(merged) > 1:
        merged[-1] = total_ms
    else:
        merged = [0, total_ms]
    return merged


def segment_audio(filepath: str, silence_thresh: int = -40,
                  min_silence_len: int = 500, min_segment_len: int = 8000,
                  progress_callback=None) -> tuple[list[dict], int]:
    audio = AudioSegment.from_mp3(filepath)
    total_ms = len(audio)

    boundaries = []

    if progress_callback:
        progress_callback("Detecting silence gaps...")
    seek_step = max(1, min_silence_len // 2)
    silent_ranges = detect_silence(audio, min_silence_len=min_silence_len,
                                   silence_thresh=silence_thresh, seek_step=seek_step)
    for sil_start, sil_end in silent_ranges:
        boundaries.append(sil_end)

    merged = _merge_boundaries(boundaries, total_ms, min_segment_len)
    segments = _bounds_to_segments(merged)
    if len(segments) > 1:
        return segments, total_ms

    if progress_callback:
        progress_callback("No clear silence gaps — using spectral analysis...")
    try:
        y, sr = librosa.load(filepath, sr=22050, mono=True)

        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_smooth = np.convolve(rms, np.ones(20) / 20, mode="same")
        low_energy_mask = rms_smooth < 0.15 * rms_smooth.max()

        in_silence = False
        sil_start_frame = 0
        lib_boundaries = []
        for i, is_silent in enumerate(low_energy_mask):
            if is_silent and not in_silence:
                in_silence = True
                sil_start_frame = i
            elif not is_silent and in_silence:
                in_silence = False
                sil_len_frames = i - sil_start_frame
                if sil_len_frames * 512 / sr * 1000 >= min_silence_len:
                    mid = (sil_start_frame + i) // 2
                    lib_boundaries.append(int(mid * 512 / sr * 1000))
        if in_silence:
            mid = (sil_start_frame + len(low_energy_mask)) // 2
            lib_boundaries.append(int(mid * 512 / sr * 1000))

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = librosa.util.peak_pick(onset_env, pre_max=5, post_max=5,
                                       pre_avg=5, post_avg=5, delta=0.3, wait=15)
        for p in peaks:
            ms = int(p * 512 / sr * 1000)
            if 0 < ms < total_ms:
                lib_boundaries.append(ms)

        merged = _merge_boundaries(lib_boundaries, total_ms, min_segment_len)
        segments = _bounds_to_segments(merged)
    except Exception:
        pass

    if len(segments) <= 1:
        if progress_callback:
            progress_callback("Splitting evenly by duration...")
        n = max(2, int(total_ms / 180000))
        chunk = total_ms / n
        merged = [0] + [int(round(i * chunk)) for i in range(1, n)] + [total_ms]
        merged = _merge_boundaries(merged[1:-1], total_ms, min_segment_len)
        segments = _bounds_to_segments(merged)

    return segments, total_ms


def _bounds_to_segments(bounds: list[int]) -> list[dict]:
    segments = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        dur = end - start
        segments.append(dict(index=len(segments) + 1, start=start, end=end, dur=dur))
    return segments


def _parse_tracklist(description: str) -> list[dict]:
    if not description:
        return []
    tracks = []
    for line in description.splitlines():
        line = line.strip()
        m = re.match(r'(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)', line)
        if m:
            parts = list(map(int, m.group(1).split(':')))
            sec = parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
            title = m.group(2).strip().rstrip('.')
            if 0 <= sec < 86400:
                tracks.append({"start_time": sec, "title": title})
    return tracks


def _assign_titles(segments: list[dict], chapters: list[dict], description: str) -> list[dict]:
    candidates = []
    if chapters:
        candidates = [{"start_time": ch["start_time"], "title": ch["title"]} for ch in chapters]
    if not candidates:
        candidates = _parse_tracklist(description)
    if not candidates:
        return segments

    for seg in segments:
        seg_start_s = seg["start"] / 1000
        best = min(candidates, key=lambda c: abs(c["start_time"] - seg_start_s))
        if abs(best["start_time"] - seg_start_s) < 15:
            seg["title"] = best["title"]
        else:
            seg["title"] = None
    return segments


def plot_waveform_and_segments(filepath: str, segments: list[dict], total_ms: int, prelude_dur_ms: int = 15000):
    audio = AudioSegment.from_mp3(filepath)
    raw = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        raw = raw.reshape(-1, audio.channels).mean(axis=1)

    sr = audio.frame_rate
    duration = len(raw) / sr
    downsample = max(1, len(raw) // 3000)
    t = np.arange(0, len(raw), downsample) / sr
    waveform = raw[::downsample].astype(np.float32)
    peak = max(abs(waveform.max()), abs(waveform.min())) or 1

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.fill_between(t, waveform / peak, -waveform / peak, alpha=0.3, color="#8b5cf6")
    ax.plot(t, waveform / peak, color="#6d3fc0", linewidth=0.5)

    colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
    for seg, color in zip(segments, colors):
        s, e = seg["start"] / 1000, seg["end"] / 1000
        ax.axvspan(s, e, alpha=0.15, color=color)

        ps = s
        pe = min(e, s + prelude_dur_ms / 1000)
        ax.axvspan(ps, pe, alpha=0.35, color=color, label=f"Prelude {seg['index']}" if seg["index"] == 1 else "")

        mid = (s + e) / 2
        title = seg.get("title")
        label = title if title else str(seg["index"])
        ax.text(mid, 0.95, label, ha="center", va="top", fontsize=9,
                fontweight="bold", color="#333",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax.set_xlim(0, duration)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Time (mm:ss)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)//60:02d}:{int(x)%60:02d}"))
    ax.set_facecolor("#f8f8ff")
    fig.patch.set_facecolor("#f8f8ff")

    legend_elements = [Rectangle((0, 0), 1, 1, color="gray", alpha=0.15, label="Song segment"),
                       Rectangle((0, 0), 1, 1, color="gray", alpha=0.35, label="Prelude (intro)")]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.8, edgecolor="#ccc")
    plt.tight_layout()
    return fig


def extract_prelude(filepath: str, start_ms: int, prelude_dur_ms: int, out_path: str) -> str:
    audio = AudioSegment.from_mp3(filepath)
    end_ms = min(start_ms + prelude_dur_ms, len(audio))
    clip = audio[start_ms:end_ms].fade_out(300)
    clip.export(out_path, format="mp3", bitrate="128k")
    return out_path


st.set_page_config(page_title="Song Prelude Extractor", page_icon="🎵", layout="wide")
st.markdown("""
<style>
    .seg-card { background:#1a1a2e; border-radius:12px; padding:16px; margin:8px 0;
                border:1px solid #2a2a5a; }
    .seg-card:hover { border-color:#8b5cf6; }
    .time-label { font-size:20px; font-weight:700; color:#8b5cf6; }
    .time-sub { font-size:13px; color:#9090b0; }
    .song-title { font-size:16px; color:#e8e8f0; font-weight:600; margin-top:6px; }
    .prelude-label { font-size:11px; color:#22c55e; font-weight:600; text-transform:uppercase;
                     letter-spacing:1px; margin-top:4px; }
    .stApp { background: #0f0f1a; }
    h1, h2, h3 { color: #e8e8f0 !important; }
</style>
""", unsafe_allow_html=True)

st.title("🎵 Song Prelude Extractor")
st.markdown("Paste a YouTube link to a video containing **multiple songs in sequence** "
            "to hear a short intro prelude of each detected song.")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...",
                    label_visibility="collapsed")

with st.expander("⚙️  Settings", expanded=False):
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        silence_thresh = st.slider("Silence threshold (dB)", -60, -20, -40,
                                   help="Lower = more sensitive to quiet sounds")
    with col_b:
        min_silence_len = st.slider("Min gap between songs (ms)", 100, 3000, 600,
                                    help="Minimum silence duration to split songs")
    with col_c:
        min_segment_len = st.slider("Min song length (s)", 3, 60, 10,
                                    help="Segments shorter than this are discarded") * 1000
    with col_d:
        prelude_sec = st.slider("Prelude duration (s)", 3, 30, 15,
                                help="How many seconds of each song intro to extract")

prelude_dur_ms = prelude_sec * 1000

if st.button("🔍  Analyze & Extract Preludes", type="primary", use_container_width=True) and url.strip():
    with st.status("Processing...", expanded=True) as status:
        st.write("📥 Downloading audio...")
        try:
            filepath, video_title, chapters, description = download_audio(url.strip())
            status.update(label=f"✅ Downloaded: {video_title}", state="running")
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

        st.write("✂️  Detecting song boundaries...")
        status_box = st.empty()
        segments, total_ms = segment_audio(
            filepath, silence_thresh=silence_thresh,
            min_silence_len=min_silence_len,
            min_segment_len=min_segment_len,
            progress_callback=lambda msg: status_box.info(msg))

        st.write("🏷️  Identifying song names...")
        _assign_titles(segments, chapters, description)
        named = sum(1 for s in segments if s.get("title"))

        st.write(f"🎧 Extracting {prelude_sec}s preludes...")
        prelude_paths = []
        combined = AudioSegment.empty()
        for seg in segments:
            seg_path = CACHE_DIR / f"prelude{seg['index']:03d}.mp3"
            extract_prelude(filepath, seg["start"], prelude_dur_ms, str(seg_path))
            prelude_paths.append(seg_path)
            clip = AudioSegment.from_mp3(str(seg_path))
            combined += clip + AudioSegment.silent(duration=500)

        medley_path = CACHE_DIR / "all_preludes.mp3"
        combined.export(str(medley_path), format="mp3", bitrate="128k")

        status.update(label=f"✅ Done — {len(segments)} songs detected ({named} identified)", state="complete")

    if not segments:
        st.warning("No song segments detected. Try adjusting the detection settings.")
        st.stop()

    total_dur_str = ms_to_str(total_ms)
    st.markdown(f"**Video:** {video_title}  ·  **Duration:** {total_dur_str}  ·  "
                f"**Songs detected:** {len(segments)}  ·  "
                f"**Prelude:** {prelude_sec}s each")

    fig = plot_waveform_and_segments(filepath, segments, total_ms, prelude_dur_ms)
    st.pyplot(fig)

    st.divider()

    st.markdown("### 🎧 All Preludes (medley)")
    with open(medley_path, "rb") as f:
        st.audio(f.read(), format="audio/mp3")

    st.divider()

    for seg, prelude_path in zip(segments, prelude_paths):
        start_str = ms_to_str(seg["start"])
        end_str = ms_to_str(seg["end"])
        dur_str = ms_to_str(seg["dur"])
        title = seg.get("title")
        title_line = f'<div class="song-title">🎵 {title}</div>' if title else ""
        with st.container():
            st.markdown(
                '<div class="seg-card">'
                '<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">'
                '<div style="background:#8b5cf6;border-radius:50%;width:40px;height:40px;'
                'display:flex;align-items:center;justify-content:center;'
                f'font-weight:800;font-size:16px">{seg["index"]}</div>'
                "<div>"
                f'<div class="time-label">{start_str} – {end_str}</div>'
                f'<div class="time-sub">Duration: {dur_str}</div>'
                f"{title_line}"
                f'<div class="prelude-label">▶ prelude ({prelude_sec}s)</div>'
                "</div></div></div>",
                unsafe_allow_html=True,
            )

            with open(prelude_path, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
