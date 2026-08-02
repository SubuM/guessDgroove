"""
guessDgroove v12 - Song Prelude Quiz
Page 1: paste a YouTube link and extract the preludes.
Choose what to guess: artist name, track/song name, or album/movie name.
Page 2: quiz — hear each song prelude, type your guess and check it.
Run with: streamlit run v12.py
Cloud: deploys as a Docker web service (e.g. Render free tier) with system
ffmpeg; static-ffmpeg is only a local fallback when ffmpeg is not installed.
"""

import os, re, io, difflib, shutil
from pathlib import Path

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yt_dlp

import static_ffmpeg
from pydub import AudioSegment
from pydub.silence import detect_silence

_ffmpeg_path = None
try:
    static_ffmpeg.add_paths(weak=True)
    _ffmpeg_path = shutil.which("ffmpeg")
except Exception as e:
    st.warning(f"⚠️ ffmpeg could not be initialised: {e}. "
               "Audio processing may fail.")

CACHE_DIR = Path("./cache/v7")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False)
def download_audio(url: str) -> tuple:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(CACHE_DIR / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "quiet": True, "no_warnings": True,
        "noplaylist": True,
    }
    if _ffmpeg_path:
        opts["ffmpeg_location"] = _ffmpeg_path
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = str(CACHE_DIR / f"{info['id']}.mp3")
        title = info.get("title", "Unknown")
        chapters = info.get("chapters") or []
        description = info.get("description") or ""
        meta = {
            "artist": info.get("artist") or info.get("creator"),
            "track": info.get("track"),
            "album": info.get("album") or info.get("alt_title"),
            "channel": info.get("channel") or info.get("uploader"),
            "upload_date": info.get("upload_date"),
            "duration": info.get("duration"),
            "view_count": info.get("view_count"),
        }
        return filepath, title, chapters, description, meta


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


def _split_title(title: str) -> tuple:
    if not title:
        return None, None
    for sep in (" - ", " – ", " — ", " | "):
        if sep in title:
            parts = [p.strip() for p in title.split(sep) if p.strip()]
            if len(parts) >= 2:
                return parts[0], sep.join(parts[1:])
    return None, title


def _mode_target(seg: dict, meta: dict, mode: str) -> tuple:
    title = seg.get("title")
    artist, track = _split_title(title)
    if mode == "artist":
        return artist, "artist name"
    if mode == "track":
        return track or title, "track / song name"
    if mode == "album":
        return meta.get("album"), "album / movie name"
    return title, "song name"


def _normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\b(feat|ft|featuring|remix|original|radio.?edit|explicit|clean|version)\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


def check_guess(guess: str, actual: str, min_ratio: float = 0.85) -> bool:
    ng, na = _normalize(guess), _normalize(actual)
    if not ng or not na:
        return False
    if ng == na:
        return True
    if min_ratio >= 1.0:
        return False
    if ng in na or na in ng:
        return True
    return difflib.SequenceMatcher(None, ng, na).ratio() >= min_ratio


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

if "v12_stage" not in st.session_state:
    st.session_state["v12_stage"] = "upload"
if "v12_guess_min_ratio" not in st.session_state:
    st.session_state["v12_guess_min_ratio"] = 0.85
if "v12_quiz_idx" not in st.session_state:
    st.session_state["v12_quiz_idx"] = 0
if "v12_mode" not in st.session_state:
    st.session_state["v12_mode"] = "track"

st.title("🎵 Song Prelude Quiz")
st.markdown("Paste a YouTube link to a video containing **multiple songs in sequence** "
            "to hear a short intro prelude of each detected song.")

if st.session_state["v12_stage"] == "upload":
    url = st.text_input("YouTube URL", key="v12_url",
                        placeholder="https://www.youtube.com/watch?v=...",
                        label_visibility="collapsed")

    with st.expander("⚙️  Settings", expanded=False):
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            silence_thresh = st.slider("Silence threshold (dB)", -60, -20, -40,
                                       help="Lower = more sensitive to quiet sounds")
        with col_b:
            min_silence_len = st.slider("Min gap between songs (ms)", 100, 3000, 100,
                                        help="Minimum silence duration to split songs")
        with col_c:
            min_segment_len = st.slider("Min song length (s)", 3, 60, 60,
                                        help="Segments shorter than this are discarded") * 1000
        with col_d:
            prelude_sec = st.slider("Prelude duration (s)", 3, 30, 15,
                                    help="How many seconds of each song intro to extract")
        guess_min_ratio = st.slider("Guess check strictness", 0.50, 1.00, 0.85, 0.05,
                                    help="Minimum similarity score needed for a guess to count as correct. "
                                         "1.00 = exact match only (after normalizing case/punctuation). "
                                         "Lower = more lenient.")
        st.session_state["v12_guess_min_ratio"] = guess_min_ratio

    prelude_dur_ms = prelude_sec * 1000

    if st.button("🔍  Analyze & Extract Preludes", type="primary", use_container_width=True) and url.strip():
        with st.status("Processing...", expanded=True) as status:
            st.write("📥 Downloading audio...")
            try:
                filepath, video_title, chapters, description, meta = download_audio(url.strip())
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

            status.update(label=f"✅ Done — {len(segments)} songs detected", state="complete")

        if not segments:
            st.warning("No song segments detected. Try adjusting the detection settings.")
            st.stop()

        st.session_state["v12_results"] = {
            "url": url.strip(),
            "filepath": filepath,
            "video_title": video_title,
            "chapters": chapters,
            "description": description,
            "meta": meta,
            "segments": segments,
            "total_ms": total_ms,
            "prelude_paths": [str(p) for p in prelude_paths],
            "medley_path": str(medley_path),
            "prelude_dur_ms": prelude_dur_ms,
        }

    results = st.session_state.get("v12_results")
    if results and results["url"] == url.strip():
        meta = results.get("meta", {})
        segments = results["segments"]

        has_artist = bool(meta.get("artist")) or any(
            _split_title(s.get("title"))[0] for s in segments)
        has_track = bool(meta.get("track")) or any(
            (_split_title(s.get("title"))[1] or s.get("title")) for s in segments)
        has_album = bool(meta.get("album"))

        st.success(f"✅ Analysis complete — **{len(segments)} songs detected**. "
                   "Choose what to guess:")
        avail = [lbl for lbl, ok in (("artist name", has_artist),
                                     ("track / song name", has_track),
                                     ("album / movie name", has_album)) if ok]
        if avail:
            st.caption("Available: " + ", ".join(avail) + ". "
                       "Disabled buttons mean that data isn't present in the video "
                       "metadata or in the chapter/tracklist titles.")
        else:
            st.caption("No artist/track/album data found in this video's metadata "
                       "or chapters — nothing can be graded.")

        c1, c2, c3 = st.columns(3)
        with c1:
            artist_help = ("Type the artist name for each prelude."
                           if has_artist else
                           "Not available — no artist found in the video metadata "
                           "or chapter/tracklist titles.")
            if st.button("🎤  Artist name", type="primary",
                         use_container_width=True, disabled=not has_artist,
                         help=artist_help):
                st.session_state["v12_mode"] = "artist"
                st.session_state["v12_quiz_idx"] = 0
                st.session_state["v12_stage"] = "quiz"
                st.rerun()
        with c2:
            track_help = ("Type the track / song name for each prelude."
                          if has_track else
                          "Not available — no chapter/tracklist titles found "
                          "for the detected songs.")
            if st.button("🎵  Track / Song name", type="primary",
                         use_container_width=True, disabled=not has_track,
                         help=track_help):
                st.session_state["v12_mode"] = "track"
                st.session_state["v12_quiz_idx"] = 0
                st.session_state["v12_stage"] = "quiz"
                st.rerun()
        with c3:
            album_help = ("Type the album / movie name for each prelude."
                          if has_album else
                          "Not available — no album/movie info in the video metadata.")
            if st.button("💿  Album / Movie name", type="primary",
                         use_container_width=True, disabled=not has_album,
                         help=album_help):
                st.session_state["v12_mode"] = "album"
                st.session_state["v12_quiz_idx"] = 0
                st.session_state["v12_stage"] = "quiz"
                st.rerun()

else:
    results = st.session_state.get("v12_results")
    if not results:
        st.warning("No analysis found. Go back and analyze a video first.")
        if st.button("←  Back to upload"):
            st.session_state["v12_stage"] = "upload"
            st.rerun()
        st.stop()

    segments = results["segments"]
    prelude_dur_ms = results["prelude_dur_ms"]
    prelude_sec = prelude_dur_ms // 1000
    prelude_paths = [Path(p) for p in results["prelude_paths"]]
    video_title = results["video_title"]
    meta = results.get("meta", {})
    mode = st.session_state.get("v12_mode", "track")
    mode_labels = {"artist": "artist name", "track": "track / song name",
                   "album": "album / movie name"}
    mode_emoji = {"artist": "🎤", "track": "🎵", "album": "💿"}
    guess_min_ratio = st.session_state["v12_guess_min_ratio"]

    st.markdown(f"### {mode_emoji.get(mode, '🎧')}  Quiz — {video_title}")
    st.markdown(f"**{len(segments)} songs detected** · One prelude per page · "
                f"Guess: **{mode_labels.get(mode, 'song name')}** · "
                f"Strictness: **{guess_min_ratio:.2f}**")

    col_top = st.columns(2)
    with col_top[0]:
        if st.button("←  New upload"):
            st.session_state["v12_stage"] = "upload"
            st.session_state["v12_quiz_idx"] = 0
            st.rerun()
    with col_top[1]:
        st.markdown(f"<p style='text-align:right;color:#9090b0'>Song "
                    f"{st.session_state['v12_quiz_idx'] + 1} of {len(segments)}</p>",
                    unsafe_allow_html=True)

    st.divider()

    quiz_idx = st.session_state["v12_quiz_idx"]
    if quiz_idx >= len(segments):
        st.success("🎉  Quiz complete!")
        st.markdown("### 📊  Summary")

        rows = []
        for i, seg in enumerate(segments):
            idx = seg["index"]
            target, target_label = _mode_target(seg, meta, mode)
            guess = st.session_state.get(f"v12_guess_{idx}", "").strip()
            checked = st.session_state.get(f"v12_checked_{idx}", False)
            if not checked:
                status_icon, status_label = "⏭️", "Skipped"
            elif not target:
                status_icon, status_label = "➖", "Not graded"
            elif check_guess(guess, target, min_ratio=guess_min_ratio):
                status_icon, status_label = "✅", "Correct"
            else:
                status_icon, status_label = "❌", "Wrong"
            rows.append((i, idx, target, target_label, guess, status_icon, status_label, checked))

        score = sum(1 for r in rows if r[6] == "Correct")
        answered = sum(1 for r in rows if r[7])
        st.markdown(f"**Score: {score} / {len(segments)} correct** ({answered} answered, "
                    f"{len(segments) - answered} skipped)")
        st.progress(score / len(segments))
        st.divider()

        for i, idx, target, target_label, guess, icon, status_label, checked in rows:
            t = target if target else f"({target_label} unavailable)"
            g = f"*{guess}*" if guess else "—"
            st.markdown(f"{icon}  **{i + 1}. {t}**  <span style='color:#9090b0'>· "
                        f"your guess: {g} · {status_label}</span>",
                        unsafe_allow_html=True)

        st.divider()
        if st.button("🔄  Play again"):
            st.session_state["v12_quiz_idx"] = 0
            st.rerun()
        if st.button("🏠  Back to upload"):
            st.session_state["v12_stage"] = "upload"
            st.session_state["v12_quiz_idx"] = 0
            st.rerun()
        st.stop()

    seg = segments[quiz_idx]
    prelude_path = prelude_paths[quiz_idx]
    start_str = ms_to_str(seg["start"])
    end_str = ms_to_str(seg["end"])
    dur_str = ms_to_str(seg["dur"])
    idx = seg["index"]

    target, target_label = _mode_target(seg, meta, mode)

    guess_key = f"v12_guess_{idx}"
    checked_key = f"v12_checked_{idx}"

    if guess_key not in st.session_state:
        st.session_state[guess_key] = ""
    if checked_key not in st.session_state:
        st.session_state[checked_key] = False

    with st.container():
        st.markdown(
            '<div class="seg-card">'
            '<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">'
            '<div style="background:#8b5cf6;border-radius:50%;width:40px;height:40px;'
            'display:flex;align-items:center;justify-content:center;'
            f'font-weight:800;font-size:16px">{idx}</div>'
            "<div>"
            f'<div class="time-label">{start_str} – {end_str}</div>'
            f'<div class="time-sub">Duration: {dur_str}</div>'
            f'<div class="prelude-label">▶ prelude ({prelude_sec}s)</div>'
            "</div></div></div>",
            unsafe_allow_html=True,
        )

        with open(prelude_path, "rb") as f:
            st.audio(f.read(), format="audio/mp3")

        col_inp, col_btn = st.columns([3, 1])
        with col_inp:
            st.text_input(f"Your guess ({target_label})", key=guess_key,
                         disabled=st.session_state[checked_key],
                         placeholder=f"Type {target_label}...")
        with col_btn:
            if st.button("Check", key=f"v12_btn_{idx}",
                        disabled=st.session_state[checked_key],
                        use_container_width=True):
                st.session_state[checked_key] = True
                st.rerun()

        if st.session_state[checked_key]:
            guess = st.session_state[guess_key]
            if target and guess.strip():
                correct = check_guess(guess, target, min_ratio=guess_min_ratio)
                if correct:
                    st.success(f"✅ Correct! → **{target}**")
                else:
                    st.error(f"❌ Wrong! Actual: **{target}**")
            elif target:
                st.warning(f"Answer: **{target}**")
            else:
                st.info(f"No {target_label} available for this segment")

    st.divider()

    nav_cols = st.columns(2)
    with nav_cols[0]:
        if quiz_idx > 0 and st.button("←  Previous", use_container_width=True):
            st.session_state["v12_quiz_idx"] = quiz_idx - 1
            st.rerun()
    with nav_cols[1]:
        if quiz_idx < len(segments) - 1:
            if st.button("Next →", key="v12_next", type="primary",
                         use_container_width=True):
                st.session_state["v12_quiz_idx"] = quiz_idx + 1
                st.rerun()
        else:
            if st.button("Finish ✅", key="v12_finish", type="primary",
                         use_container_width=True):
                st.session_state["v12_quiz_idx"] = len(segments)
                st.rerun()
