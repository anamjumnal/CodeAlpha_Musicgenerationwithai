from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import uuid
import traceback
import json

import numpy as np
import scipy.io.wavfile
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration


# ============================================================
# SOUNDFORGE
# ============================================================

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent

GENERATED_DIR = BASE_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)

LIBRARY_FILE = BASE_DIR / "library.json"


# ============================================================
# ORIGINAL 4 TRACKS
# THESE ALWAYS STAY SEPARATE FROM MY LIBRARY
# ============================================================

ORIGINAL_TRACKS = [
    {
        "id": 1,
        "title": "Generated Track 01",
        "genre": "Classical",
        "icon": "🎹",
        "file": "generated_music_1_loud.wav",
        "duration": "3:45"
    },
    {
        "id": 2,
        "title": "Generated Track 02",
        "genre": "Classical",
        "icon": "🎻",
        "file": "generated_music_2_loud.wav",
        "duration": "3:52"
    },
    {
        "id": 3,
        "title": "Generated Track 03",
        "genre": "Classical",
        "icon": "🎼",
        "file": "generated_music_3_loud.wav",
        "duration": "3:58"
    },
    {
        "id": 4,
        "title": "Generated Track 04",
        "genre": "Classical",
        "icon": "🎺",
        "file": "generated_music_4_loud.wav",
        "duration": "4:02"
    }
]


# ============================================================
# LIBRARY STORAGE
# ============================================================

def load_library():
    if not LIBRARY_FILE.exists():
        return []

    try:
        with open(
            LIBRARY_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except Exception:
        traceback.print_exc()

    return []


def save_library(tracks):
    try:
        with open(
            LIBRARY_FILE,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                tracks,
                f,
                indent=4,
                ensure_ascii=False
            )

        return True

    except Exception:
        traceback.print_exc()
        return False


# ============================================================
# MUSICGEN
# ============================================================

MODEL_NAME = "facebook/musicgen-small"

processor = None
model = None


def load_musicgen():
    global processor, model

    if processor is not None and model is not None:
        return

    print()
    print("=" * 70)
    print("SOUNDFORGE - LOADING MUSICGEN")
    print("=" * 70)
    print("Model:", MODEL_NAME)
    print("First load can take some time.")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_NAME
    )

    model.to("cpu")
    model.eval()

    print("=" * 70)
    print("MUSICGEN LOADED SUCCESSFULLY")
    print("=" * 70)
    print()


# ============================================================
# GENERATE MUSIC
# ============================================================

def generate_music_audio(
    prompt,
    genre,
    creativity,
    requested_duration
):

    requested_duration = int(
        max(
            5,
            min(
                int(requested_duration),
                300
            )
        )
    )

    sampling_rate = int(
        model.config.audio_encoder.sampling_rate
    )

    target_samples = (
        sampling_rate * requested_duration
    )

    full_prompt = (
        f"{genre} instrumental music. "
        f"{prompt}. "
        f"No vocals. "
        f"High quality musical composition. "
        f"Natural dynamics, musical progression, "
        f"clear melody and coherent arrangement."
    )

    print()
    print("=" * 70)
    print("NEW SOUNDFORGE AI GENERATION")
    print("=" * 70)
    print("Genre:", genre)
    print("Prompt:", full_prompt)
    print(
        "Requested duration:",
        requested_duration,
        "seconds"
    )
    print(
        "Creativity:",
        creativity
    )
    print("=" * 70)

    chunks = []

    remaining = requested_duration
    chunk_number = 1

    while remaining > 0:

        chunk_duration = min(
            remaining,
            30
        )

        print(
            f"\nGenerating section "
            f"{chunk_number}: "
            f"{chunk_duration} seconds"
        )

        if chunk_number == 1:

            chunk_prompt = full_prompt

        else:

            chunk_prompt = (
                full_prompt
                +
                " Continue the same instrumental "
                "composition naturally with variation "
                "and development. Do not simply repeat "
                "the previous section."
            )

        inputs = processor(
            text=[chunk_prompt],
            padding=True,
            return_tensors="pt"
        )

        max_new_tokens = (
            int(
                np.ceil(
                    (chunk_duration + 1) * 50
                )
            )
            + 60
        )

        print(
            "Generating approximately",
            chunk_duration,
            "seconds..."
        )

        with torch.no_grad():

            audio_values = model.generate(
                **inputs,
                do_sample=True,
                temperature=creativity,
                guidance_scale=3.0,
                max_new_tokens=max_new_tokens
            )

        audio = (
            audio_values[0]
            .cpu()
            .numpy()
        )

        audio = np.squeeze(audio)

        if audio.ndim > 1:
            audio = audio[0]

        audio = audio.astype(
            np.float32
        )

        target_chunk_samples = (
            int(
                sampling_rate
                * chunk_duration
            )
        )

        audio = audio[
            :target_chunk_samples
        ]

        chunks.append(audio)

        remaining -= chunk_duration
        chunk_number += 1

    # ========================================================
    # JOIN CHUNKS
    # ========================================================

    final_audio = chunks[0]

    for next_chunk in chunks[1:]:

        crossfade_samples = min(
            int(
                sampling_rate * 0.5
            ),
            len(final_audio),
            len(next_chunk)
        )

        if crossfade_samples <= 0:

            final_audio = np.concatenate(
                [
                    final_audio,
                    next_chunk
                ]
            )

            continue

        first_main = final_audio[
            :-crossfade_samples
        ]

        first_tail = final_audio[
            -crossfade_samples:
        ]

        second_head = next_chunk[
            :crossfade_samples
        ]

        second_rest = next_chunk[
            crossfade_samples:
        ]

        fade_out = np.linspace(
            1.0,
            0.0,
            crossfade_samples,
            dtype=np.float32
        )

        fade_in = np.linspace(
            0.0,
            1.0,
            crossfade_samples,
            dtype=np.float32
        )

        blended = (
            first_tail * fade_out
            +
            second_head * fade_in
        )

        final_audio = np.concatenate(
            [
                first_main,
                blended,
                second_rest
            ]
        )

    # ========================================================
    # EXACT LENGTH
    # ========================================================

    final_audio = final_audio[
        :target_samples
    ]

    # ========================================================
    # NORMALIZE
    # ========================================================

    peak = np.max(
        np.abs(final_audio)
    )

    if peak > 0:
        final_audio = (
            final_audio / peak
        ) * 0.95

    final_audio = final_audio.astype(
        np.float32
    )

    print(
        "Final duration:",
        f"{len(final_audio) / sampling_rate:.2f}s"
    )

    return (
        final_audio,
        sampling_rate
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        BASE_DIR,
        "SoundForge.html"
    )


# ============================================================
# CSS
# ============================================================

@app.route("/style.css")
def css():

    return send_from_directory(
        BASE_DIR,
        "style.css"
    )


# ============================================================
# JAVASCRIPT
# ============================================================

@app.route("/script.js")
def javascript():

    return send_from_directory(
        BASE_DIR,
        "script.js"
    )


# ============================================================
# ORIGINAL AUDIO
# ============================================================

@app.route("/audio/<path:filename>")
def original_audio(filename):

    filename = Path(
        filename
    ).name

    # First check project root
    root_file = BASE_DIR / filename

    if root_file.exists():

        return send_from_directory(
            BASE_DIR,
            filename,
            mimetype="audio/wav"
        )

    # Then check audio folder
    audio_dir = BASE_DIR / "audio"

    audio_file = (
        audio_dir / filename
    )

    if audio_file.exists():

        return send_from_directory(
            audio_dir,
            filename,
            mimetype="audio/wav"
        )

    return jsonify({
        "success": False,
        "error": "Original audio file not found.",
        "filename": filename
    }), 404


# ============================================================
# GENERATED AUDIO
# ============================================================

@app.route("/generated/<path:filename>")
def generated_audio(filename):

    filename = Path(
        filename
    ).name

    file_path = (
        GENERATED_DIR / filename
    )

    if not file_path.exists():

        return jsonify({
            "success": False,
            "error": "Generated audio file not found."
        }), 404

    return send_from_directory(
        GENERATED_DIR,
        filename,
        mimetype="audio/wav"
    )


# ============================================================
# LIBRARY
# ============================================================

@app.route(
    "/api/library",
    methods=["GET"]
)
def get_library():

    tracks = load_library()

    valid_tracks = []

    for track in tracks:

        filename = Path(
            track.get(
                "file",
                ""
            )
        ).name

        file_path = (
            GENERATED_DIR / filename
        )

        if file_path.exists():
            valid_tracks.append(track)

    if len(valid_tracks) != len(tracks):

        save_library(
            valid_tracks
        )

    return jsonify({
        "success": True,
        "tracks": valid_tracks
    })


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "online": True,

        "project":
            "SoundForge",

        "model":
            MODEL_NAME,

        "model_loaded":
            model is not None,

        "original_tracks":
            len(ORIGINAL_TRACKS),

        "library_tracks":
            len(load_library()),

        "ai_generation":
            True,

        "duration_support":
            "5-300 seconds"
    })


# ============================================================
# ORIGINAL TRACK API
# ============================================================

@app.route(
    "/api/tracks",
    methods=["GET"]
)
def get_tracks():

    tracks = []

    for track in ORIGINAL_TRACKS:

        tracks.append({
            **track,
            "file":
                "/audio/"
                +
                track["file"]
        })

    return jsonify({
        "success": True,
        "tracks": tracks
    })


# ============================================================
# AI GENERATION
# ============================================================

@app.route(
    "/api/generate",
    methods=["POST"]
)
def generate_music():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        prompt = str(
            data.get(
                "prompt",
                ""
            )
        ).strip()

        genre = str(
            data.get(
                "genre",
                ""
            )
        ).strip()

        try:

            requested_duration = int(
                float(
                    data.get(
                        "duration",
                        10
                    )
                )
            )

        except (
            TypeError,
            ValueError
        ):

            requested_duration = 10

        requested_duration = max(
            5,
            min(
                requested_duration,
                300
            )
        )

        try:

            creativity = float(
                data.get(
                    "creativity",
                    0.8
                )
            )

        except (
            TypeError,
            ValueError
        ):

            creativity = 0.8

        creativity = max(
            0.5,
            min(
                creativity,
                1.5
            )
        )

        if not prompt:

            return jsonify({
                "success": False,
                "error":
                    "Please enter a music prompt."
            }), 400

        if not genre:

            return jsonify({
                "success": False,
                "error":
                    "Please choose a genre."
            }), 400

        # ====================================================
        # LOAD MODEL
        # ====================================================

        load_musicgen()

        # ====================================================
        # GENERATE
        # ====================================================

        audio, sampling_rate = (
            generate_music_audio(
                prompt,
                genre,
                creativity,
                requested_duration
            )
        )

        # ====================================================
        # SAVE FILE
        # ====================================================

        track_id = uuid.uuid4().hex[:10]

        filename = (
            f"ai_generated_{track_id}.wav"
        )

        output_path = (
            GENERATED_DIR / filename
        )

        scipy.io.wavfile.write(
            str(output_path),
            sampling_rate,
            audio
        )

        if not output_path.exists():

            raise RuntimeError(
                "Generated WAV file was not created."
            )

        # ====================================================
        # AUDIO URL
        # ====================================================

        audio_url = (
            f"/generated/{filename}"
        )

        # ====================================================
        # ACTUAL DURATION
        # ====================================================

        actual_duration = (
            len(audio)
            / sampling_rate
        )

        minutes = int(
            actual_duration // 60
        )

        seconds = int(
            actual_duration % 60
        )

        duration_text = (
            f"{minutes}:{seconds:02d}"
        )

        # ====================================================
        # TRACK OBJECT
        # ====================================================

        track = {

            "id":
                track_id,

            "title":
                "AI Generated Track",

            "genre":
                genre,

            "icon":
                "🎵",

            "file":
                audio_url,

            "audio_url":
                audio_url,

            "duration":
                duration_text,

            "prompt":
                prompt,

            "created_at":
                datetime.now().isoformat()
        }

        # ====================================================
        # SAVE TO MY LIBRARY
        # ====================================================

        library = load_library()

        library.insert(
            0,
            track
        )

        save_library(
            library
        )

        # ====================================================
        # SUCCESS
        # ====================================================

        print()
        print("=" * 70)
        print("GENERATION SUCCESSFUL")
        print("=" * 70)
        print("File:", output_path)
        print(
            "Duration:",
            duration_text
        )
        print("=" * 70)

        return jsonify({

            "success":
                True,

            "message":
                "AI music generated and saved to My Library.",

            "track":
                track
        })

    except Exception as e:

        print()
        print("=" * 70)
        print("SOUNDFORGE MUSICGEN ERROR")
        print("=" * 70)

        traceback.print_exc()

        print("=" * 70)

        return jsonify({

            "success":
                False,

            "error":
                str(e)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SOUNDFORGE AI MUSIC GENERATOR")
    print("=" * 70)
    print(
        "Website:",
        "http://127.0.0.1:5000"
    )
    print(
        "AI Model:",
        MODEL_NAME
    )
    print(
        "Original tracks:",
        len(ORIGINAL_TRACKS)
    )
    print(
        "Library:",
        LIBRARY_FILE
    )
    print(
        "Generated:",
        GENERATED_DIR
    )
    print(
        "Duration support:",
        "5-300 seconds"
    )
    print("=" * 70)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
