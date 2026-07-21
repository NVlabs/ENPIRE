# Voice Input and Output

The voice system provides two capabilities for the CAP agent framework:

1. **Voice input (STT)** -- host microphone capture with realtime transcription
   streamed to the CAP chat UI via WebSocket.
2. **Voice output (TTS)** -- spoken playback of assistant responses via the
   bridge, using either a local system TTS engine (RealtimeTTS) or the
   ElevenLabs cloud API.

Both subsystems live under `cap/voice/` and are consumed by the agent bridge
(`cap/bridge/agent_bridge.py`) and the React chat panel (`cap/ui/`).

---

## Architecture overview

```
                    +------------------+
                    |   CAP Chat UI    |  (React, port 5173)
                    |  ChatPanel.tsx   |
                    +--------+---------+
                             |
              +--------------+--------------+
              | Vite proxy                   |
              |  /voice-api -> :8202/api     |
              |  /voice-ws  -> ws://:8202/ws |
              |  /bridge-api -> :8201/api    |
              |  /bridge-ws  -> ws://:8201/ws|
              +--------------+--------------+
              |                              |
   +----------v-----------+    +------------v-----------+
   |  Voice Input Server  |    |   Agent Bridge Server  |
   |  cap.voice.voice_server   |   cap.bridge.agent_bridge  |
   |  FastAPI :8202        |    |   FastAPI :8201        |
   |  VoiceInputService   |    |   ChatVoiceController  |
   |  (RealtimeSTT)       |    |   VoiceOutputManager   |
   +----------------------+    |   (RealtimeTTS or      |
                               |    ElevenLabs)          |
                               +------------------------+
```

### File map

| File | Purpose |
|------|---------|
| `cap/voice/__init__.py` | Public API: `ChatVoiceController`, `VoiceInputService`, `VoiceOutputManager`, `VoiceStatus`, `extract_speakable_text` |
| `cap/voice/voice_server.py` | Standalone FastAPI app for voice input (port 8202) |
| `cap/voice/service.py` | `VoiceInputService` -- manages the RealtimeSTT recorder lifecycle |
| `cap/voice/backends.py` | TTS backends: `SystemVoiceOutputBackend` (RealtimeTTS), `ElevenLabsVoiceOutputBackend`, `PygameMP3AudioPlayer` |
| `cap/voice/output.py` | `VoiceOutputManager` -- queued async TTS worker |
| `cap/voice/chat_output.py` | `ChatVoiceController` -- accumulates assistant text deltas and speaks on turn completion |
| `cap/voice/speakable_text.py` | `extract_speakable_text()` -- strips markdown/code to produce speakable prose |
| `cap/voice/elevenlabs_account.py` | `list_elevenlabs_voices()`, `pick_recommended_female_english_voice()` helpers |
| `cap/config.py:278-279` | `CAP_VOICE_HOST` and `CAP_VOICE_PORT` constants |
| `cap/bridge/agent_bridge.py` | Bridge server that creates `ChatVoiceController`, wires TTS to chat turns, proxies voice-input stop |
| `cap/chat/runtime.py` | Generic chat runtime also consuming `ChatVoiceController` for voice output |
| `cap/ui/src/hooks/useChatStream.ts` | React hook: voice WebSocket connection, `VoiceState` management |
| `cap/ui/src/components/ChatPanel.tsx` | Chat panel: mic controls, transcript bar, send-transcript button |
| `cap/ui/src/api/client.ts:204-239` | Bridge voice REST helpers: `voiceStatus()`, `setVoiceEnabled()`, `voiceTest()`, `voiceSpeak()`, `voiceStop()` |
| `cap/ui/vite.config.ts:30-39` | Vite dev proxy: `/voice-api` -> `:8202/api`, `/voice-ws` -> `ws://:8202/ws` |
| `tools/voice/debug_active_mic.py` | Diagnostic CLI: enumerates PipeWire/PulseAudio sources, runs a live RMS meter, optionally saves a WAV capture |
| `tools/teleop_voice_annotate/voice_annotation.py` | Teleop voice annotation thread for data collection (trigger-word temporal alignment) |
| `tools/teleop_voice_annotate/visualize_annotations.py` | Renders voice annotation text onto multi-camera episode videos |
| `third_party/ui/RealtimeSTT/` | Vendored RealtimeSTT library (fallback if pip package unavailable) |
| `third_party/RealtimeTTS/` | Vendored RealtimeTTS library used by `SystemVoiceOutputBackend` |

---

## Install

### Voice input (STT) dependencies

```bash
uv sync --extra stt
```

This installs (from `pyproject.toml:101-113`):

- `PyAudio 0.2.14`
- `faster-whisper 1.1.1` (Whisper model backend)
- `webrtcvad-wheels` (WebRTC VAD)
- `pvporcupine` (wake-word)
- `openwakeword`
- `RealtimeSTT` (from git)

Linux audio system packages:

```bash
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio alsa-utils
```

### Voice output (TTS) dependencies

The `SystemVoiceOutputBackend` requires RealtimeTTS (vendored under
`third_party/RealtimeTTS/`). The `ElevenLabsVoiceOutputBackend` has no
additional Python dependencies beyond the stdlib (uses `urllib.request` and
`pygame` for audio playback).

---

## Voice input (STT)

### Voice input server

**Entry point**: `cap/voice/voice_server.py` (run as `uv run cap/voice/voice_server.py`)

The server is a standalone FastAPI application (`cap/voice/voice_server.py:55-109`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/start` | POST | Start microphone capture and transcription |
| `/api/stop` | POST | Stop listening |
| `/api/clear` | POST | Clear partial/final transcript text |
| `/api/status` | GET | Return current voice state snapshot |
| `/ws` | WebSocket | Realtime event stream |

**Default port**: `8202` (env `CAP_VOICE_PORT`, defined at `cap/config.py:279`).

#### WebSocket events

All events are JSON with the shape `{"type": str, "data": dict, "timestamp": str}`.

| Event type | Data fields | Emitted when |
|------------|-------------|--------------|
| `voice_status` | `listening`, `preloading`, `preloaded`, `phase`, `partial_text`, `final_text`, `error` | Any state change |
| `voice_partial` | `text` | Realtime transcription update (while speaking) |
| `voice_final` | `text` | Utterance finalized after silence detection |
| `voice_error` | `error` | Recorder error |

#### VoiceInputService lifecycle

Defined at `cap/voice/service.py:21-295`.

**Phase state machine**:

```
idle -> preloading -> ready -> starting -> listening -> recording -> processing -> transcribing -> finalized -> listening (loop)
                                        -> error
                          -> stopping -> ready / idle
```

- **Preloading** (`service.py:85-99`): On server startup (`@app.on_event("startup")`
  at `voice_server.py:101-103`), `VoiceInputService.preload()` builds a
  `RealtimeSTT.AudioToTextRecorder` in a background thread to warm up the
  Whisper model. The recorder is immediately shut down after preload --
  the purpose is to download/cache the model weights.
- **Listening** (`service.py:265-295`): `_run()` creates a fresh recorder and
  enters a blocking loop calling `recorder.text(callback)`. RealtimeSTT fires
  callbacks for partial transcription, recording start/stop, and transcription
  start, which the service relays as WebSocket events.
- **Recorder config** (`service.py:204-236`): Uses `model="tiny.en"`, realtime
  transcription enabled with `realtime_model_type="tiny.en"`,
  `silero_sensitivity=0.05`, `webrtc_sensitivity=3`,
  `post_speech_silence_duration=0.25`, `beam_size=1`.

### Frontend voice input

The React UI connects to the voice input server via WebSocket at `/voice-ws`
(proxied to `ws://localhost:8202/ws` by Vite at `vite.config.ts:35-39`).

**Hook**: `useChatStream` (`cap/ui/src/hooks/useChatStream.ts:61-445`)

- Maintains `VoiceState` (`useChatStream.ts:45-51`): `listening`, `phase`,
  `partialText`, `finalText`, `error`.
- `connectVoice()` (`useChatStream.ts:132-184`): Opens the WebSocket, fetches
  initial status via REST, reconnects on close (up to 10 retries).
- `startVoiceListening()` (`useChatStream.ts:393-400`): Suppresses stale voice
  events, clears transcript, stops any bridge TTS, then POSTs `/voice-api/start`.
- `stopVoiceListening()` (`useChatStream.ts:402-404`): POSTs `/voice-api/stop`.
- `dismissVoiceCapture()` (`useChatStream.ts:417-422`): Suppresses events,
  resets state, stops and clears voice server.

**UI Component**: `ChatPanel` (`cap/ui/src/components/ChatPanel.tsx:26-308`)

The chat panel renders a voice control bar (`ChatPanel.tsx:239-279`) containing:

- **Start/Stop mic** button: toggles `onVoiceStart` / `onVoiceStop`.
- **Send transcript** button: sends `voice.finalText` (or `voice.partialText`)
  as a chat message.
- **Clear** button: resets transcript state.
- **Transcript bar** (`data-testid="voice-transcript-bar"`): shows realtime
  partial text, final text, or error.
- When final text arrives, it auto-populates the text input field
  (`ChatPanel.tsx:50-53`).

---

## Voice output (TTS)

### Backend selection

Controlled by env `CAP_VOICE_OUTPUT_PROVIDER` (or `CAP_VOICE_PROVIDER`),
resolved in `cap/voice/backends.py:318-326`:

| Provider value | Backend class | Description |
|----------------|---------------|-------------|
| `system` (default) | `SystemVoiceOutputBackend` | Local TTS via `RealtimeTTS.SystemEngine` |
| `elevenlabs` | `ElevenLabsVoiceOutputBackend` | ElevenLabs cloud API + pygame playback |

### SystemVoiceOutputBackend

Defined at `cap/voice/backends.py:158-190`. Uses `RealtimeTTS.TextToAudioStream`
with `SystemEngine()` for local system TTS. Requires PipeWire/PulseAudio audio
environment (env vars `XDG_RUNTIME_DIR`, `DBUS_SESSION_BUS_ADDRESS` set
automatically at `backends.py:18-22`).

### ElevenLabsVoiceOutputBackend

Defined at `cap/voice/backends.py:295-316`.

**Configuration** (`ElevenLabsConfig.from_env()` at `backends.py:117-155`):

| Env var | Default | Description |
|---------|---------|-------------|
| `ELEVENLABS_API_KEY` | (required) | API key (also accepts `ELVENSLAB_API_KEY`) |
| `ELEVENLABS_VOICE_ID` | (required) | Voice to use |
| `ELEVENLABS_MODEL_ID` | `eleven_flash_v2_5` | TTS model |
| `ELEVENLABS_LANGUAGE_CODE` | `en` | Language code |
| `ELEVENLABS_OUTPUT_FORMAT` | `mp3_44100_128` | Audio format |
| `ELEVENLABS_VOICE_PRESET` | `conversational` | Preset (`default` or `conversational`) |
| `ELEVENLABS_STABILITY` | from preset | Voice stability |
| `ELEVENLABS_SIMILARITY_BOOST` | from preset | Similarity boost |
| `ELEVENLABS_STYLE` | from preset | Style |
| `ELEVENLABS_SPEED` | from preset | Speed |
| `ELEVENLABS_USE_SPEAKER_BOOST` | from preset | Speaker boost |
| `ELEVENLABS_TIMEOUT_S` | `30.0` | Request timeout |

Audio playback uses `PygameMP3AudioPlayer` (`backends.py:231-293`) which loads
the MP3 bytes from ElevenLabs into `pygame.mixer.music`.

### VoiceOutputManager

Defined at `cap/voice/output.py:23-123`. A queued TTS worker that:

1. Receives text via `speak_async(text)` (`output.py:50-58`).
2. Passes through `extract_speakable_text()` to strip markdown/code.
3. Queues the cleaned text for a background worker thread.
4. The worker calls `backend.speak(text)` synchronously.

### ChatVoiceController

Defined at `cap/voice/chat_output.py:7-47`. A thin controller that:

- Accumulates streaming text deltas via `on_text_delta(text)` (`chat_output.py:14-16`).
- On `on_turn_complete()` (`chat_output.py:18-21`), joins all accumulated text
  and queues it for TTS via `VoiceOutputManager.speak_async()`.
- On `on_turn_error()` (`chat_output.py:23-24`), discards accumulated text.

### Speakable text extraction

`extract_speakable_text()` (`cap/voice/speakable_text.py:15-32`) strips fenced
code blocks, inline code, markdown links, headers, bullet points, numbered
lists, bold/italic markers, and pipes/em-dashes. Produces clean prose suitable
for TTS.

### Bridge voice output integration

The agent bridge (`cap/bridge/agent_bridge.py:340-349`) creates a
`ChatVoiceController` wrapping a `VoiceOutputManager`.

- Voice output enabled by default; disable with `CAP_BRIDGE_VOICE_ENABLED=0`.
- Auto-disabled in pytest (`agent_bridge.py:341-342`).
- Before speaking, the bridge calls `_stop_voice_input_capture()`
  (`agent_bridge.py:90-98`) to POST `/api/stop` to the voice input server,
  preventing mic feedback.
- `_TurnWSManager` (`agent_bridge.py:292-303`) intercepts `chat_text_delta`
  events to feed `chat_voice.on_text_delta()`.

Bridge voice REST endpoints (`agent_bridge.py:420-447`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/voice/status` | GET | Get TTS status (enabled, speaking, last text, error) |
| `/api/voice/enabled` | POST | Enable/disable voice output |
| `/api/voice/test` | POST | Speak a test phrase (default "Hello World") |
| `/api/voice/speak` | POST | Speak arbitrary text |
| `/api/voice/stop` | POST | Stop current TTS playback |

---

## ElevenLabs account helpers

`cap/voice/elevenlabs_account.py` provides:

- `list_elevenlabs_voices(api_key)` (`elevenlabs_account.py:40-74`): Lists all
  voices on the account via `/v1/voices`.
- `pick_recommended_female_english_voice(voices)` (`elevenlabs_account.py:77-97`):
  Scores and ranks voices, preferring "Rachel", female, English, conversational.
- `format_voice_summary(voice)` (`elevenlabs_account.py:100-113`): Format for display.
- Default voice ID: `21m00Tcm4TlvDq8ikWAM` (Rachel) at `elevenlabs_account.py:14`.

---

## Teleop voice annotation (data collection)

Separate from the CAP chat voice system, `tools/teleop_voice_annotate/` provides
voice-annotated data collection for teleoperation episodes.

### voice_annotation.py

`tools/teleop_voice_annotate/voice_annotation.py:52-189`

`start_voice_annotation_thread(env, ...)` starts a daemon thread that:

1. Creates a `RealtimeSTT.AudioToTextRecorder` with `model="tiny.en"`.
2. Listens continuously; on transcription, applies **trigger-word alignment**:
   - Start triggers: `"start"`, `"begin"`, `"now"` (`voice_annotation.py:27`).
   - End triggers: `"done"`, `"finish"`, `"stop"` (`voice_annotation.py:28`).
   - Speech between a start and end trigger is buffered and attached to the
     recording frame range.
3. In non-trigger mode (`trigger_mode=False`), all speech is recorded directly.
4. Returns `(thread, stop_event, apply_cached_label)`.

### visualize_annotations.py

`tools/teleop_voice_annotate/visualize_annotations.py` renders annotation text
as a dark banner overlaid on stacked multi-camera episode videos (top | left | right).
Loads annotations from `*_annotation.json`, writes `annotated_video.mp4`.

---

## Microphone debug tool

`tools/voice/debug_active_mic.py` is a standalone diagnostic script:

```bash
python tools/voice/debug_active_mic.py --configure
python tools/voice/debug_active_mic.py --configure --seconds 8 --save test.wav
```

What it does (`debug_active_mic.py:8-22`):

1. Sets PipeWire/PulseAudio audio env vars.
2. Finds and prints the default PipeWire source with ALSA card details.
3. Optionally unmutes and raises volume (`--configure`).
4. Enumerates PyAudio input devices and auto-selects the best candidate.
5. Runs a live RMS/peak audio meter.
6. Optionally saves a WAV capture.

---

## Running

### Full stack (all-in-one launcher)

```bash
tmux/cap_agent_interface/start_cap_agent_interface.sh
```

This launches (`start_cap_agent_interface.sh:91-131`):

1. `cap.server.cap_server` on `:8300`
2. `cap.agent.cap_agent` on `:8200`
3. `cap.bridge.claude_bridge` (agent bridge) on `:8201`
4. `cap.voice.voice_server` (voice input) on `:8202`
5. Vite UI dev server on `:5173`

Stop all:

```bash
tmux/cap_agent_interface/stop_cap_agent_interface.sh
```

Check status:

```bash
tmux/cap_agent_interface/status_cap_agent_interface.sh
```

### Individual services

```bash
# Voice input server
uv run -m cap.voice.voice_server

# Agent bridge (includes voice output)
uv run -m cap.bridge.claude_bridge

# UI dev server
cd cap/ui && npm run dev
```

The voice input server listens on port `8202` by default.

### CAP UI flow

1. Open the CAP UI at `http://localhost:5173`.
2. Switch to **Agent** mode (the chat panel appears).
3. Click **Start mic** in the voice control bar.
4. Speak into the host microphone.
5. Watch realtime transcript updates in the transcript bar below the mic button.
6. Click **Send transcript** to submit the finalized text to the agent chat.
7. The assistant response is automatically spoken via TTS (if voice output is
   enabled).
8. Click **Voice Test** in the header to test TTS with "Hello World".

---

## Network ports and proxy configuration

| Service | Port | Env var |
|---------|------|---------|
| Voice input server | 8202 | `CAP_VOICE_PORT` |
| Agent bridge | 8201 | `BRIDGE_PORT` |
| Vite UI | 5173 | -- |

Vite proxy routes (`cap/ui/vite.config.ts:11-39`):

| UI path | Target |
|---------|--------|
| `/voice-api/*` | `http://localhost:8202/api/*` |
| `/voice-ws/*` | `ws://localhost:8202/ws/*` |
| `/bridge-api/*` | `http://localhost:8201/api/*` |
| `/bridge-ws/*` | `ws://localhost:8201/ws/*` |
| `/api/*` | `http://localhost:8200/api/*` |
| `/ws/*` | `ws://localhost:8200/ws/*` |

---

## Environment variables reference

| Variable | Default | Used by | Description |
|----------|---------|---------|-------------|
| `CAP_VOICE_PORT` | `8202` | voice_server, config.py | Voice input server port |
| `CAP_VOICE_HOST` | `localhost` | config.py, agent_bridge | Voice input server host |
| `CAP_VOICE_API_URL` | `http://{host}:{port}/api` | agent_bridge | Full voice input API URL override |
| `CAP_VOICE_OUTPUT_PROVIDER` | `system` | backends.py | TTS backend: `system` or `elevenlabs` |
| `CAP_VOICE_PROVIDER` | `system` | backends.py | Alias for the above |
| `CAP_BRIDGE_VOICE_ENABLED` | `1` | agent_bridge | Enable/disable bridge voice output |
| `ELEVENLABS_API_KEY` | -- | backends.py | ElevenLabs API key (required for elevenlabs backend) |
| `ELEVENLABS_VOICE_ID` | -- | backends.py | ElevenLabs voice ID (required for elevenlabs backend) |
| `ELEVENLABS_MODEL_ID` | `eleven_flash_v2_5` | backends.py | ElevenLabs model |
| `ELEVENLABS_LANGUAGE_CODE` | `en` | backends.py | ElevenLabs language |
| `ELEVENLABS_OUTPUT_FORMAT` | `mp3_44100_128` | backends.py | ElevenLabs output format |
| `ELEVENLABS_VOICE_PRESET` | `conversational` | backends.py | ElevenLabs voice preset |

---

## Tests

### Backend tests

```bash
# Voice input server (unit)
uv run pytest tests/test_voice_server.py

# Voice output manager
uv run pytest tests/test_voice_output.py

# ChatVoiceController
uv run pytest tests/test_chat_voice_controller.py

# Bridge voice API endpoints
uv run pytest tests/test_claude_bridge_voice_api.py

# ElevenLabs API integration (live, requires API key)
RUN_ELEVENLABS_API_VOICE_TEST=1 uv run pytest tests/test_elevenslab_api_voice.py

# Shell script for UI voice output manual test
bash tests/run_voice_output_ui_test.sh
```

### Frontend tests

```bash
cd cap/ui

# Voice input tests
npm test -- ChatPanel.voice

# Voice output tests
npm test -- ChatPanel.voiceOutput
```

Test files:

- `tests/test_voice_server.py` -- voice input server REST/WebSocket
- `tests/test_voice_output.py` -- `VoiceOutputManager` and backends
- `tests/test_chat_voice_controller.py` -- `ChatVoiceController` accumulation/speak
- `tests/test_claude_bridge_voice_api.py` -- bridge `/api/voice/*` endpoints
- `tests/test_elevenslab_api_voice.py` -- live ElevenLabs TTS integration
- `cap/ui/src/__tests__/ChatPanel.voice.test.tsx` -- voice input UI controls
- `cap/ui/src/__tests__/ChatPanel.voiceOutput.test.tsx` -- voice output UI

---

## Quickstart manual test (voice input only)

You do not need the full CAP stack to verify the voice input module.

Terminal 1:

```bash
uv sync --extra stt
uv run -m cap.voice.voice_server
```

Terminal 2:

```bash
cd cap/ui
npm install
npm run dev
```

Then open `http://localhost:5173`, switch to Agent mode, and click **Start mic**.

---

## Cross-references

- **CAP system architecture**: `docs/CAP_DESIGN.md`
- **CAP UI layout and components**: `docs/CAP_UI_DESIGN.md`
- **Remote serving (port assignments)**: `docs/remote_serving.md`
- **Data collection annotations**: `docs/DATA_STUDIO.md`
- **Table bussing launch scripts**: `docs/TABLE_BUSSING_SKILLS.md`
