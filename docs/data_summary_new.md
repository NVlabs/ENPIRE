# Data Loader Survey for Online Robot Foundation Model Training

> **Goal**: Design an online training loop where rollout experiences are collected in real-time and streamed into a growing training dataset on CPU, with efficient disk-to-RAM data loading for large-scale VLA/world model training.

---

## Table of Contents

1. [LeRobot (HuggingFace)](#1-lerobot-huggingface)
2. [Open X-Embodiment / RLDS](#2-open-x-embodiment--rlds)
3. [OpenVLA](#3-openvla)
4. [Open-Pi-Zero (pi0 re-implementation)](#4-open-pi-zero)
5. [Robo-DM (Berkeley)](#5-robo-dm-berkeley)
6. [RLinf / RLinf-USER](#6-rlinf--rlinf-user)
7. [Evo-RL (MINT-SJTU)](#7-evo-rl-mint-sjtu)
8. [CR-DAgger (Stanford)](#8-cr-dagger-stanford)
9. [Protobuf for Robotics Data](#9-protobuf-for-robotics-data)
10. [MCAP (Foxglove)](#10-mcap-foxglove)
11. [Comparative Summary Table](#11-comparative-summary-table)
12. [Recommendations for Our Pipeline](#12-recommendations-for-our-pipeline)
13. [YAM Platform: Current Data Collection Implementation](#13-yam-platform-current-data-collection-implementation)
    - [13.1 Data Collection Entry Points](#131-data-collection-entry-points)
    - [13.2 Policy Interface](#132-policy-interface)
    - [13.3 Gymnasium Environment](#133-gymnasium-environment)
    - [13.4 Episode Recording: RecordEpisodeWrapper](#134-episode-recording-recordepisodewrapper)
    - [13.5 Episode Data Format (on disk)](#135-episode-data-format-on-disk)
    - [13.6 Metadata Schema](#136-metadata-schema-metadatajson)
    - [13.7 Data Replay](#137-data-replay)
    - [13.8 Timing Instrumentation](#138-timing-instrumentation)
    - [13.9 Camera Pipeline](#139-camera-pipeline)
    - [13.10 IPC and Communication](#1310-ipc-and-communication)
    - [13.11 Data Format vs External Systems](#1311-data-format-vs-external-systems)
    - [13.12 Tmux Launch Architecture](#1312-tmux-launch-architecture)

---

## 1. LeRobot (HuggingFace)

**Repo**: https://github.com/huggingface/lerobot  
**Format version**: v3.0 (lerobot >= 0.4.0), v2.1 (current stable)

### Data Format

LeRobot v3.0 uses a **file-based** storage model (multiple episodes packed per file):

| Layer | Format | Purpose |
|-------|--------|---------|
| Tabular data | Apache Parquet shards | States, actions, timestamps — memory-mapped via PyArrow |
| Visual data | MP4 shards per camera | Multiple episodes concatenated per file |
| Metadata | JSON + Parquet | Schema, FPS, normalization stats, episode boundaries |

Directory layout:
```
dataset/
├── videos/           # MP4 shards per camera (many episodes per file)
├── data/             # Frame-by-frame Parquet shards
├── meta/
│   ├── episodes/     # Per-episode records (chunked Parquet)
│   ├── tasks.jsonl   # Task descriptions → integer IDs
│   ├── stats.json    # Global feature statistics
│   └── info.json     # Schema, FPS, version, path templates
```

### Data Loading Classes

1. **`LeRobotDataset`** — Standard map-style PyTorch Dataset. Returns `dict[str, Tensor]`. Supports `delta_timestamps` for temporal windowing (e.g., observation history). Wraps with standard `torch.utils.data.DataLoader`.

2. **`StreamingLeRobotDataset`** — Streams directly from HuggingFace Hub without downloading. Uses HF `datasets` library streaming + `torchvision.io.VideoReader`. Good for federated / distributed training.

3. **`LeRobotDatasetV2` (internal, evolved from `OnlineBuffer`)** — Backed by `numpy.memmap` instead of PyArrow. ~10x faster iteration than `datasets.Dataset`. Supports in-place updates critical for online RL replay buffers.

### Online Training Support

- `add_frame()` API for frame-level data ingestion during rollouts
- `save_episode()` + `finalize()` for persisting collected episodes to disk
- The v2 `OnlineBuffer`→`LeRobotDatasetV2` migration specifically targets online training: numpy.memmap enables overwriting old data efficiently in a fixed-size buffer

### Strengths for Our Use Case

- **Mature ecosystem** — direct integration with HF Hub, standard PyTorch DataLoader
- **Parquet + mmap** for efficient random access to tabular data
- **StreamingLeRobotDataset** enables training without full dataset download
- **Active development** — v3.0 designed for millions of episodes at scale

### Weaknesses

- MP4 video decoding is a bottleneck for high-throughput training (50x slower than Robo-DM)
- `save_episode()` involves full episode buffering before write
- No native asynchronous data collection / training decoupling
- Memory issue reported: eager loading during `save_episode_table` can cause OOM on large datasets (issue #1346)

---

## 2. Open X-Embodiment / RLDS

**Repo**: https://github.com/google-deepmind/open_x_embodiment  
**Scale**: 1M+ real trajectories, 22 embodiments, 527 skills, ~4.5 TB

### Data Format

Uses **RLDS (Robot Learning Dataset Standard)** — episode-structured format built on TensorFlow Datasets (TFDS). Each dataset is a sequence of episodes stored as TFRecords.

### PyTorch Data Loading Options

| Loader | Type | Streaming | Notes |
|--------|------|-----------|-------|
| `tfds.load()` (official) | TF-native | Yes | Canonical but requires TF |
| `OpenXExperienceReplay` (TorchRL) | PyTorch IterableDataset | Yes | Part of torchrl 0.7+, supports trajectory sampling |
| `IterableOpenXDataset` (community) | PyTorch IterableDataset | Yes | Regex-based filtering, configurable sample lengths |
| `OpenXDataset` (community) | PyTorch map-style | No | Full in-memory loading |

### Strengths

- Largest standardized robot dataset collection
- TorchRL integration provides replay buffer primitives
- Streaming support for RAM-constrained machines

### Weaknesses

- TFRecord format is not PyTorch-native — requires bridges
- No native online data ingestion API
- Heavy dependency on Google Cloud Storage for data access

---

## 3. OpenVLA

**Repo**: https://github.com/openvla/openvla  
**Model**: 7B-param VLA, trained on 970k demonstrations

### Data Loading

- Natively consumes **RLDS format** datasets from Open X-Embodiment
- Training powered by **PyTorch FSDP** + Flash-Attention
- Supports 1B–34B models via fully sharded data parallel
- FAST tokenizer compresses action chunks into fewer tokens (15x faster inference)

### Online Training Relevance

OpenVLA focuses on offline pre-training and fine-tuning, not online RL. Fine-tuning on LIBERO and other benchmarks uses standard offline data loading. The data pipeline is RLDS→PyTorch with no dedicated online buffer.

---

## 4. Open-Pi-Zero

**Repo**: https://github.com/allenzren/open-pi-zero  
**Model**: 3B PaliGemma + 0.315B action expert

### Data Pipeline

- Uses **RLDS datasets** (Bridge, Fractal) for training
- Custom `VLAProcessor` for data preprocessing
- Action/proprioception normalized to [-1, 1]
- Flow matching loss on action chunks (chunk size 4)
- Training: batch 16, gradient accumulation 8, ~1.5–2 days on L40

### Data Format Convention

Documents dataset-specific conventions for proprioceptive formats (quaternion vs. Euler) and gripper representations across different robot platforms.

### Relevance

Standard offline RLDS loading — no online data streaming infrastructure. Useful as a reference for VLA input/output format conventions.

---

## 5. Robo-DM (Berkeley)

**Repo**: https://github.com/BerkeleyAutomation/fog_x (also https://github.com/BerkeleyAutomation/robodm)  
**Paper**: CoRL 2024 / ICRA 2025

### Data Format

Uses **EBML (Extensible Binary Meta Language)** with custom `.vla` file extension.

### Key Performance Numbers

| Metric | Robo-DM | vs. RLDS | vs. LeRobot |
|--------|---------|----------|-------------|
| Compression (lossy) | Up to 70x | 70x better | — |
| Compression (lossless) | Up to 3.5x | 3.5x better | — |
| Data retrieval speed | — | — | **50x faster** |
| Downstream accuracy | No degradation at 75x compression | — | — |

### Architecture

```python
import robodm

# Write — append-only, streaming-friendly
trajectory = robodm.Trajectory(path="demo.vla", mode="w")
for step in range(N):
    trajectory.add("camera/rgb", image)
    trajectory.add("robot/joint_positions", joints)
    trajectory.add("action/gripper_action", action)
trajectory.close()

# Read — lazy loading with memory-mapped decode cache
trajectory = robodm.Trajectory(path="demo.vla", mode="r")
data = trajectory.load()
```

### Key Features

- **Video codec flexibility**: rawvideo, FFV1 (lossless), H.264, H.265, AV1 (best compression)
- **Load-balancing video decoding** with memory-mapped decode caches
- **Format interop**: native conversion to/from Open-X-Embodiment, HuggingFace, RLDS, HDF5
- **PyTorch integration**: `pip install -e .[torch]`
- **Distributed ready**: flexible dataset partitioning

### Strengths for Our Use Case

- **Fastest retrieval** of any surveyed system (50x faster than LeRobot)
- **Streaming-friendly** append-only write path
- Excellent compression preserves downstream task accuracy
- Native conversion to/from LeRobot format

### Weaknesses

- Younger project, less community adoption than LeRobot
- No built-in online RL buffer management
- Actively refactoring (their README notes heavy code restructuring)

---

## 6. RLinf / RLinf-USER

**Repo**: https://github.com/RLinf/RLinf  
**Paper**: arXiv:2602.07837 (RLinf-USER), arXiv:2509.15965 (RLinf system)

### Architecture: Fully Asynchronous Online RL Pipeline

This is the most directly relevant system for our online training loop design.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  EnvWorker   │────▶│   Distributed │────▶│  TrainWorker │
│  (rollouts)  │     │   Channel     │     │  (GPU)       │
└─────────────┘     │   (buffer)    │     └─────────────┘
                     └──────────────┘            │
                           ▲                     │ weight sync
                           │                     ▼
                     ┌──────────────┐     ┌─────────────┐
                     │  Persistent   │     │  Inference   │
                     │  Cache-Aware  │     │  Worker      │
                     │  Buffer       │     └─────────────┘
                     └──────────────┘
```

### Key Data Pipeline Components

1. **Worker / WorkerGroup abstractions** — distributed execution across nodes
2. **Distributed Channel** — inter-worker communication for streaming data exchange
3. **Persistent cache-aware buffer** — crash recovery, historical data reuse for long-horizon experiments
4. **Adaptive communication plane** — tunneling-based networking, streaming-multiprocessor-aware weight sync

### Performance (vs. synchronous baselines)

| Component | Speedup |
|-----------|---------|
| Data generation throughput | 1.20–1.55× |
| Training throughput | **4.61–5.70×** |

### Supported Algorithms & Models

- **RL**: SAC, RLPD, CrossQ, SAC-Flow, DSRL, PPO, GRPO
- **Imitation**: HG-DAgger, Full-param SFT, LoRA SFT
- **Models**: π₀, π₀.₅, OpenVLA, OpenVLA-OFT, GR00T, CNN/MLP policies
- **World models**: OpenSora, Wan (RL fine-tuning of VLA via world models)

### Strengths for Our Use Case

- **Most complete online RL infrastructure** — exactly what we need
- **Asynchronous decoupling** of data collection and training is the right paradigm
- **Persistent buffer with crash recovery** — critical for long real-world experiments
- Supports VLA post-training (π₀, OpenVLA) via RL — our target scenario
- Built on top of veRL (Volcano Engine RL) — mature distributed RL backend

### Weaknesses

- Complex system with heavy dependencies (Docker recommended)
- Not natively integrated with LeRobot data format
- Primary focus on simulation environments; real-world deployment still maturing
- Large codebase — integration may require significant adaptation

---

## 7. Evo-RL (MINT-SJTU)

**Repo**: https://github.com/MINT-SJTU/Evo-RL  
**Platforms**: SO-101, AgileX PiPER/PiPER-X

### Architecture

**LeRobot-aligned foundation**: uses LeRobot as the base codebase because its inference and data-collection logic align with real-world RL workflows.

### 7-Stage Training Pipeline

1. Installation
2. Hardware setup
3. **Data collection** (teleoperation via LeRobot CLI)
4. **Value function training** (offline, from collected data)
5. **Value inference** (label existing data with learned values)
6. **Policy training** (offline RL with value-labeled data)
7. **Closed-loop rollout** → collect more data → repeat

### Data Pipeline

- Inherits LeRobot's data format (Parquet + MP4)
- Data collection via `lerobot-record` / `lerobot-teleoperate`
- Iterative offline RL: collect → label → train → rollout → collect more
- Open datasets and checkpoints for reproducibility

### Strengths

- Direct LeRobot compatibility — minimal format conversion
- Complete real-world RL pipeline on affordable hardware
- Iterative data collection loop is a practical online-ish paradigm

### Weaknesses

- Not truly online (batch offline RL with iterative data collection)
- No asynchronous training/collection decoupling
- Limited to small-scale robots (SO-101, PiPER)

---

## 8. CR-DAgger (Stanford)

**Repo**: https://github.com/yifan-hou/cr-dagger  
**Paper**: NeurIPS 2025

### Online Data Collection Approach

CR-DAgger implements a **compliant intervention interface** for on-policy corrections:

1. Base policy executes on robot
2. Human provides gentle delta corrections via compliant kinesthetic interface
3. System records: robot commands + correction episodes + force/torque data (6-axis)
4. **Residual policy** trained on correction data (15D: 9D SE(3) delta pose + 6D wrenches)

### Data Pipeline

- On-policy data collection: corrections recorded during policy execution
- Admittance controller (~1000 N/m stiffness) fuses robot + human inputs
- Trains residual policy on top of frozen base policy
- Requires minimal correction data for 50–60% success rate improvement

### Strengths

- Elegant on-policy correction paradigm reduces distribution shift
- Force/torque data enriches the training signal
- Residual policy architecture avoids catastrophic forgetting

### Weaknesses

- No standardized data format documented
- Focused on DAgger (imitation), not RL
- No replay buffer or streaming infrastructure

---

## 9. Protobuf for Robotics Data

### Overview

Protocol Buffers provide compact binary serialization with strong typing and schema enforcement. Used widely in robotics (ROS 2/DDS, MCAP, gRPC services).

### Performance Characteristics

| Property | Protobuf | JSON | FlatBuffers |
|----------|----------|------|-------------|
| Serialization speed | Fast | Slow | Fastest (zero-copy) |
| Message size | Small (~3-10x smaller than JSON) | Large | Smallest |
| Schema enforcement | Strong (compiled) | None | Strong |
| Forward/backward compat | Yes | N/A | Yes |
| Human readability | No (binary) | Yes | No |

### Relevance to Our Pipeline

**Protobuf is best suited as a wire format for inter-process communication**, not as a primary training data storage format. In our architecture:

- **Good for**: Serializing rollout observations between robot driver → data collector → buffer
- **Good for**: RPC messages (gRPC / Portal RPC already uses similar patterns)
- **Not ideal for**: Bulk training data storage (Parquet/Arrow/EBML are better for columnar access)

### Verdict

Use Protobuf (or similar schema-enforced binary format) for the **transport layer** between robot and training loop. Use Parquet/Arrow/EBML for the **storage layer**.

---

## 10. MCAP (Foxglove)

**Repo**: https://github.com/foxglove/mcap  
**Docs**: https://mcap.dev

### Format Design

MCAP is an open-source, self-describing container format for timestamped heterogeneous data:

- **Append-only, row-oriented** — fast writes under high-volume conditions
- **Built-in indexing** — efficient seeking and time-range queries
- **Compression**: LZ4, Zstandard (zstd) — configurable per-chunk
- **Serialization-agnostic**: wraps Protobuf, ROS 1/2, JSON, CDR, or raw bytes
- **Partial recovery** if recording is interrupted

### Performance

| Metric | Value |
|--------|-------|
| Go reader throughput | Up to 660 MB/s (bare lexer) |
| Ordered iteration | ~239 MB/s (without index) |
| Message rate | 21M+ messages/sec |
| Rust write latency | 173–330 µs/message |

### Python API

```python
from mcap.reader import McapReader
from mcap.writer import Writer

# Write
with open("recording.mcap", "wb") as f:
    writer = Writer(f)
    writer.start()
    # ... add channels, write messages with timestamps ...
    writer.finish()

# Read — streaming
from mcap.stream_reader import StreamReader
reader = StreamReader(open("recording.mcap", "rb"))
for record in reader.records:
    # process record
    pass

# Read — indexed (fast seeking)
reader = McapReader(open("recording.mcap", "rb"))
for schema, channel, message in reader.iter_messages(
    topics=["/camera", "/joints"],
    start_time=t0, end_time=t1
):
    # decode message based on schema
    pass
```

### ROS 2 Integration

MCAP is the **default storage format for ROS 2** (since Iron release). `ros2 bag record` produces `.mcap` files automatically.

### Relevance to Our Pipeline

**MCAP excels as a recording/logging format** — ideal for:
- Recording raw rollout data from the robot (cameras, joint states, actions, rewards)
- Time-synchronized multi-modal data capture
- Efficient append-only writes during online data collection
- Post-hoc visualization with Foxglove

**Not ideal as a direct training data format** because:
- Row-oriented (poor for columnar random access needed by DataLoaders)
- No built-in concept of "episodes" or "trajectories"
- Message-level granularity adds overhead for batch tensor loading
- Would need conversion to Parquet/Arrow for efficient training

### Recommended Role

Use MCAP as the **recording/ingestion layer** (robot → disk), then convert to Parquet/LeRobot format for the **training layer** (disk → GPU). This two-stage pipeline gives you fast recording + fast training.

---

## 11. Comparative Summary Table

| System | Storage Format | Online Buffer | Async Pipeline | Streaming | PyTorch Native | LeRobot Compat | Best For |
|--------|---------------|---------------|----------------|-----------|----------------|----------------|----------|
| **LeRobot v3** | Parquet + MP4 | numpy.memmap (v2) | No | Yes (Hub) | Yes | ✅ Native | Standard training |
| **RLDS/OXE** | TFRecord | No | No | Yes (TFDS) | Via bridge | Via converter | Large offline datasets |
| **OpenVLA** | RLDS | No | No | Via RLDS | Yes (FSDP) | Via converter | VLA fine-tuning |
| **Open-Pi-Zero** | RLDS | No | No | No | Yes | No | VLA reference impl |
| **Robo-DM** | EBML (.vla) | No | No | Yes | Yes | Via converter | **Fastest retrieval** |
| **RLinf-USER** | Custom buffer | ✅ Persistent | ✅ Full async | Yes | Yes | No | **Online RL at scale** |
| **Evo-RL** | LeRobot format | No | No | Via LeRobot | Yes | ✅ Native | Real-world iterative RL |
| **CR-DAgger** | Custom | Implicit | No | No | Yes | No | On-policy corrections |
| **Protobuf** | Binary messages | N/A | N/A | N/A | N/A | N/A | Wire format / RPC |
| **MCAP** | Container | N/A | N/A | Yes | Via reader | No | **Recording/logging** |

---

## 12. Recommendations for Our Pipeline

### Proposed Architecture: 3-Layer Data Pipeline

```
Layer 1: RECORDING (Robot → Disk)
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Robot HW   │────▶│  Protobuf/   │────▶│  MCAP or raw    │
│  (cameras,   │ UDP │  Portal RPC  │     │  Parquet append  │
│   joints)    │     │  serialized  │     │  (disk)          │
└─────────────┘     └──────────────┘     └─────────────────┘

Layer 2: BUFFER (Disk ↔ RAM)
┌─────────────────┐     ┌──────────────────────┐
│  Episode         │────▶│  numpy.memmap         │
│  Converter       │     │  Replay Buffer        │
│  (MCAP→Parquet)  │     │  (CPU RAM, mmap'd)    │
└─────────────────┘     └──────────────────────┘
                               │
                               ▼ PyTorch DataLoader
Layer 3: TRAINING (RAM → GPU)
┌──────────────────────┐
│  DataLoader           │
│  (prefetch, pin_mem)  │
│  → GPU training loop  │
└──────────────────────┘
```

### Specific Recommendations

#### 1. Adopt LeRobot v3 as the canonical training data format

**Why**: It's the emerging standard for robot learning. Parquet for tabular data + MP4 for video is well-tested. The `StreamingLeRobotDataset` and PyArrow memory-mapping give us efficient disk→RAM streaming. Evo-RL already validates that LeRobot works for iterative RL pipelines.

**Action**: Store all offline and historical data in LeRobot v3 format.

#### 2. Build an async online buffer inspired by RLinf-USER

**Why**: RLinf-USER's asynchronous architecture (4.6–5.7x training speedup) is the right paradigm. Decoupling data collection from training is essential for real-time robot operation.

**Action**: Implement a `Worker`-based architecture where:
- **EnvWorker** runs the rollout policy and writes experiences to a shared buffer
- **TrainWorker** samples from the buffer and runs gradient updates
- Communication via shared memory (`multiprocessing.shared_memory`) or numpy.memmap files
- Weight sync via periodic checkpoint writes (not blocking the rollout)

#### 3. Use numpy.memmap for the online replay buffer (LeRobot v2 approach)

**Why**: 10x faster than PyArrow for the in-place overwrite pattern needed by replay buffers. Direct memory-mapping means the OS handles disk↔RAM paging automatically — no explicit loading code needed.

**Implementation sketch**:
```python
import numpy as np

class OnlineReplayBuffer:
    def __init__(self, capacity, obs_shape, action_dim, path="/tmp/buffer"):
        self.images = np.memmap(f"{path}/images.dat", dtype=np.uint8,
                                mode='w+', shape=(capacity, *obs_shape))
        self.states = np.memmap(f"{path}/states.dat", dtype=np.float32,
                                mode='w+', shape=(capacity, state_dim))
        self.actions = np.memmap(f"{path}/actions.dat", dtype=np.float32,
                                 mode='w+', shape=(capacity, action_dim))
        self.rewards = np.memmap(f"{path}/rewards.dat", dtype=np.float32,
                                  mode='w+', shape=(capacity,))
        self.write_idx = 0
        self.size = 0

    def add(self, obs, state, action, reward):
        idx = self.write_idx % self.capacity
        self.images[idx] = obs
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.write_idx += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "images": torch.from_numpy(self.images[indices].copy()),
            "states": torch.from_numpy(self.states[indices].copy()),
            "actions": torch.from_numpy(self.actions[indices].copy()),
            "rewards": torch.from_numpy(self.rewards[indices].copy()),
        }
```

#### 4. Use MCAP or Protobuf-serialized messages for the recording layer (optional)

**When to use MCAP**: If you need time-synchronized multi-modal recording with post-hoc visualization in Foxglove. Good if the robot system already uses ROS 2.

**When to use raw Protobuf**: If you want a lightweight wire format for Portal RPC messages between the robot driver and the data collector process.

**When to skip both**: If you're already using Portal RPC with a custom binary format, adding MCAP/Protobuf may be unnecessary overhead. The key bottleneck is video decoding, not message serialization.

#### 5. Consider Robo-DM for large-scale offline datasets

**Why**: 50x faster data retrieval than LeRobot, excellent compression without accuracy loss. If you accumulate massive amounts of offline demonstration data, Robo-DM's EBML format is significantly more efficient.

**Trade-off**: Less ecosystem support than LeRobot. Best used as a complementary storage backend for large historical datasets, while keeping LeRobot as the primary format.

#### 6. Video decoding is the real bottleneck — plan for it

All surveyed systems struggle with video decoding throughput. Strategies:
- **Robo-DM's approach**: Memory-mapped video decode caches (decode once, reuse)
- **LeRobot v2 approach**: Store images as numpy.memmap instead of MP4 (10x faster)
- **Pre-decode pipeline**: Background process continuously decodes upcoming episodes
- **Resolution trade-off**: Lower resolution (224×224) is 4x faster to decode than 480×640

### Priority Ranking for Implementation

| Priority | Component | Reference System |
|----------|-----------|------------------|
| **P0** | numpy.memmap replay buffer with circular overwrite | LeRobot v2 OnlineBuffer |
| **P0** | Async data collection ↔ training decoupling | RLinf-USER |
| **P1** | LeRobot v3 format for persistent episode storage | LeRobot v3, Evo-RL |
| **P1** | PyTorch DataLoader with prefetch + pin_memory | Standard PyTorch |
| **P2** | Video decode cache / pre-decode pipeline | Robo-DM |
| **P2** | Protobuf schema for rollout messages | Portal RPC integration |
| **P3** | MCAP recording for visualization/debugging | Foxglove ecosystem |
| **P3** | Robo-DM for large historical dataset storage | Robo-DM EBML format |

---

## Appendix A: External Source Code Pointers

| Project | Key Data Files | Notes |
|---------|---------------|-------|
| LeRobot | `lerobot/common/datasets/lerobot_dataset.py` | `LeRobotDataset`, `add_frame()`, `save_episode()` |
| LeRobot | `lerobot/datasets/streaming_dataset.py` | `StreamingLeRobotDataset` |
| RLinf | `rlinf/workers/` | Worker/WorkerGroup abstractions |
| RLinf | `rlinf/data/` | Distributed Channel, replay buffer |
| Robo-DM | `robodm/trajectory.py` | `Trajectory` class, EBML read/write |
| Evo-RL | Built on LeRobot | Same data format/collection CLI |
| CR-DAgger | `cr-dagger/` | Correction data recording, residual policy |
| MCAP | `mcap` PyPI package | `McapReader`, `Writer`, `StreamReader` |
| Open-Pi-Zero | `src/openpi/data/` | RLDS loading, VLAProcessor |
| OpenVLA | `prismatic/vla/datasets/` | RLDS → PyTorch bridge |

## Appendix B: YAM Codebase File Reference

| File | Key Classes / Functions | Purpose |
|------|------------------------|---------|
| `policy.py` | `Policy` (ABC), type aliases | Base policy interface |
| `teleop_policy.py:251` | `TeleopPolicy`, `LeaderRobotClient`, `FollowerRobotClient` | Bimanual teleoperation via YAM/Fello leaders |
| `robot/fello/fello_teleop_policy.py:45` | `FelloTeleopPolicy`, `DualFelloTeleopPolicy` | Fello-specific 7-DOF leader mapping |
| `record_episode_wrapper.py:35` | `RecordEpisodeWrapper` | Gym wrapper: episode recording to disk |
| `tools/vision/run_data_collection.py:54` | `DataCollectionConfig`, `main()` | Primary data collection entry point and loop |
| `launch.py:36` | `TmuxSession`, `main()` | Multi-process tmux orchestrator |
| `robot/yam/_base_yam_env.py:17` | `_BaseYamEnv` | Base gym env: obs/action spaces, kinematics |
| `robot/yam/yam_real_env.py:101` | `YamRealEnv`, `NonBlockingCamera` | Real hardware env: Portal RPC, camera threads |
| `robot/yam/yam_sim_env.py:20` | `YamSimEnv` | MuJoCo simulation env |
| `experimental/lerobot_replay_policy.py:28` | `LerobotReplayPolicy` | Multi-format episode replay |
| `experimental/yam_control_loop.py:295` | `EvalConfig`, evaluation main loop | Eval/replay with optional recording |
| `cap/server/cap_server.py:426` | `_SkillDataRecorder` | RL skill learning data recorder |
| `timing_jsonl.py:9` | `TimingJsonlLogger` | Persistent JSONL timing instrumentation |
| `tools/teleop_voice_annotate/voice_annotation.py` | `start_voice_annotation_thread()` | Real-time speech-to-text annotation |
| `tools/teleop_voice_annotate/visualize_annotations.py` | Annotation video renderer | Post-collection annotated video generation |
| `scripts/bench_station_timing.py` | Cross-host timing benchmark | Isolates host-side timing jitter |
| `third_party/overlay_viz/app.py` | FastAPI server (40+ endpoints) | Data Studio backend (see DATA_STUDIO.md) |
| `third_party/overlay_viz/scanner.py` | Episode scanner, frame cache | Episode discovery and video frame extraction |

---

## 13. YAM Platform: Current Data Collection Implementation

This section documents how data collection, recording, replay, and storage are implemented in the YAM bimanual robot codebase. All file paths are relative to the repository root.

> **Cross-references:**
> - [DATA_STUDIO.md](DATA_STUDIO.md) — Episode browsing, replay, labeling UI
> - [debug_shit_data_collection_infra.md](debug_shit_data_collection_infra.md) — Timing investigation and instrumentation
> - [RL_PIPELINE_DESIGN.md](RL_PIPELINE_DESIGN.md) — RL training pipeline (serve_rl_policy, learn_skill)
> - [CAP_DESIGN.md](CAP_DESIGN.md) — CAP system architecture (includes skill data recording)
> - [VOICE_INPUT.md](VOICE_INPUT.md) — Voice annotation subsystem

### 13.1 Data Collection Entry Points

There are three distinct data collection pathways in the codebase:

#### 13.1.1 Teleoperation Data Collection (primary)

**Launch command:**
```bash
uv run launch.py --mode=data_collection --use-fello
uv run launch.py --mode=data_collection --use-fello --force-feedback
uv run launch.py --mode=data_collection --use-fello --use-voice
```

**Files involved:**
- `launch.py:123-173` — TmuxSession orchestrator; spawns follower servers, leader servers, camera servers, and main data collection script
- `tools/vision/run_data_collection.py` (alias: `run_data_collection.py`) — Main data collection loop
- `teleop_policy.py:251-541` — `TeleopPolicy` class: mirrors leader arm positions to follower arms
- `robot/fello/fello_teleop_policy.py:45-383` — `FelloTeleopPolicy` / `DualFelloTeleopPolicy`: 7-DOF Fello leader to YAM follower mapping
- `record_episode_wrapper.py:35-463` — `RecordEpisodeWrapper`: gym wrapper that records episodes to disk
- `tools/teleop_voice_annotate/voice_annotation.py` — Optional real-time speech-to-text annotation thread

**Configuration:**
- `tools/vision/run_data_collection.py:54-110` — `DataCollectionConfig` dataclass with all CLI flags
- Key parameters: `operator`, `station`, `use_fello`, `force_feedback`, `policy_control_freq` (default 30 Hz), `use_voice`, `voice_trigger_mode`, `timing_debug`

**Main loop flow** (`tools/vision/run_data_collection.py:337-424`):
```
while True:
    1. action, policy_info = policy.get_action(obs)      # Read leader arms
    2. action["__action_t"] = time.time()                 # Stamp action creation
    3. obs, _, _, _, step_info = env.step(action)          # Command followers + pace + read obs
    4. (optional) display images / voice annotation
    5. Check policy_info for button events: start / save / discard
```

**Button bindings (Fello mode)** — `teleop_policy.py:350-359`:
- Right pedal 0 → save episode
- Right pedal 1 → discard episode
- Right pedal 2 → start recording

#### 13.1.2 Evaluation / Policy Rollout Recording

**Launch command:**
```bash
uv run experimental/yam_control_loop.py --record-episode [policy flags]
```

**Files involved:**
- `experimental/yam_control_loop.py:295` — `record_episode` flag on `EvalConfig`
- `experimental/yam_control_loop.py:714-725` — Wraps gym env with `RecordEpisodeWrapper`, output under `$YAM_RAW_PATH/eval/`
- Uses the same `RecordEpisodeWrapper` as teleoperation
- Supports `AsyncChunkingPolicy`, `SyncChunkingPolicy`, `RealtimeRTCChunkingPolicy`, `LerobotReplayPolicy`, `HILPolicyWrapper`, `ScriptedPolicy`, `PicoPolicy`

#### 13.1.3 CAP Server Skill Learning (RL data collection)

**Files involved:**
- `cap/server/cap_server.py:426-627` — `_SkillDataRecorder` class
- Records the same directory layout as `RecordEpisodeWrapper` but from the CAP control loop
- Adds extra fields: `is-human-action.npy` (per-step takeover flag), `reward.npy` (placeholder reward signal)
- Background video encoding thread (`_frame_writer_loop`) to avoid control-loop overruns

### 13.2 Policy Interface

**File:** `policy.py:1-24`

```python
Observation = dict[str, Any]
Action = dict[str, np.ndarray]
Options = dict[str, Any]
Info = dict[str, Any]

class Policy(ABC):
    def reset(self) -> Info | None: ...
    def get_action(self, observation: Observation) -> tuple[Action, Info]: ...
```

**All Policy implementations:**

| Class | File | Purpose |
|-------|------|---------|
| `TeleopPolicy` | `teleop_policy.py:251` | Mirror leader arms to followers (bimanual YAM or Fello) |
| `FelloTeleopPolicy` | `robot/fello/fello_teleop_policy.py:45` | Single Fello arm teleoperation |
| `DualFelloTeleopPolicy` | `robot/fello/fello_teleop_policy.py:385` | Bimanual Fello teleoperation |
| `LerobotReplayPolicy` | `experimental/lerobot_replay_policy.py:28` | Replay actions from dataset (Parquet/NPY/NPZ) |
| `DatasetReplayPolicy` | `experimental/dataset_replay_policy.py:13` | Simpler 46D replay (legacy) |
| `PicoPolicy` | `experimental/pico_policy.py:22` | Pico streaming policy |
| `AsyncChunkingPolicy` | `experimental/async_chunking_policy.py` | Async action chunking for learned policies |
| `SyncChunkingPolicy` | `experimental/sync_chunking_policy.py` | Synchronous action chunking |
| `RealtimeRTCChunkingPolicy` | `experimental/realtime_rtc_chunking_policy.py` | Real-time RTC chunking |
| `HILPolicyWrapper` | `experimental/hil_policy.py` | Human-in-the-loop wrapper (teleop override) |
| `ScriptedPolicy` | `experimental/scripted_policy.py` | Scripted motion planner policy |

### 13.3 Gymnasium Environment

**Files:**
- `robot/yam/_base_yam_env.py:17` — `_BaseYamEnv(gym.Env)`: base class, observation/action spaces, kinematics, IK
- `robot/yam/yam_real_env.py:101` — `YamRealEnv(_BaseYamEnv)`: real hardware, Portal RPC to follower arm servers
- `robot/yam/yam_sim_env.py:20` — `YamSimEnv(_BaseYamEnv)`: MuJoCo simulation

**Action space** (`_base_yam_env.py:184`):

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `left_joint_pos` | (6,) | float32 | Left arm joint positions (rad) |
| `left_gripper_pos` | (1,) | float32 | Left gripper position |
| `right_joint_pos` | (6,) | float32 | Right arm joint positions (rad) |
| `right_gripper_pos` | (1,) | float32 | Right gripper position |

**Observation space** (`_base_yam_env.py:89`):
- Same joint/gripper keys as action, plus `{top,left,right}_camera_image` — (480, 640, 3) uint8

**Control loop timing** (`yam_real_env.py:224-233`):
- `policy_control_freq` (default 30 Hz) sets the pacing
- `time.sleep()` busy-wait to maintain target period
- `__timestamps` dict attached to info for per-component timing analysis

### 13.4 Episode Recording: `RecordEpisodeWrapper`

**File:** `record_episode_wrapper.py:35-463`

**Recording flow:**
1. `reset(options={"task_name": "..."})` — starts recording a new episode into a `tempfile.TemporaryDirectory`
2. `step(action)` — each step records:
   - `timestamps` — `time.time()` wall-clock (after `env.step()` returns) — line 115
   - `observations` — non-image obs from previous step (key-renamed: `left_` → `left-`) — lines 118-125
   - `actions` — strips `source` and `__action_t` metadata keys — lines 128-131
   - `action_sources` — per-step `"human"` / `"policy"` / `"unknown"` label — line 128
   - `component_timestamps` — per-component creation times (obs timestamps from env + action timestamp) — lines 133-141
   - Video frames — written to mp4v via OpenCV `VideoWriter` — lines 149-160
3. `reset(options={"start_new_episode": False})` — finalizes episode: saves all arrays and converts video to H.264
4. `reset(options={"discard_episode": True})` — discards without saving

**Finalization** (`_finalize_episode`, line 206):
- `_save_timestamps()` → `timestamp.npy`
- `_save_component_timestamps()` → `component_timestamps.json`
- `_save_observations()` → per-key `.npy` files (float64)
- `_save_actions()` → `action-left-pos.npy`, `action-right-pos.npy`, `action-source.npy`, `action-source.json`
- `_save_metadata()` → `metadata.json`
- `_save_annotations()` → `*_annotation.json`
- `_convert_videos_to_h264()` → ffmpeg re-encode (libx264, crf=23, yuv420p)
- Episode dir named: `YYYYMMDDTHHMMSS<microseconds>`

**Voice annotations** (`record_episode_wrapper.py:90-108`, `tools/teleop_voice_annotate/voice_annotation.py`):
- `add_voice_annotation(frame_idx, text)` — records segment-based annotations
- Thread-safe via `_annotations_lock`
- Trigger words: start/begin/now (start), done/finish/stop (end)
- Saved as `top_camera-images-rgb_annotation.json`

### 13.5 Episode Data Format (on disk)

**Root directory:** `$YAM_RAW_PATH` (environment variable, required)

**Directory layout:**
```
$YAM_RAW_PATH/
  {operator}_{task}_{timestamp}-YAM-{station:02d}/     # teleoperation
    {YYYYMMDDTHHMMSS<microseconds>}/                    # one episode
      ├── timestamp.npy                   # (T,) float64 — UNIX wall-clock seconds
      ├── component_timestamps.json       # [{obs_key: float, action: float, ...}, ...]
      ├── action-left-pos.npy             # (T, 7) float32 — 6 joints + 1 gripper
      ├── action-right-pos.npy            # (T, 7) float32
      ├── action-source.npy               # (T,) str — "human" | "policy" | "unknown"
      ├── action-source.json              # same, JSON list (backward compat)
      ├── left-joint_pos.npy              # (T, 6) float64 — observation
      ├── left-gripper_pos.npy            # (T, 1) float64
      ├── right-joint_pos.npy             # (T, 6) float64
      ├── right-gripper_pos.npy           # (T, 1) float64
      ├── top_camera-images-rgb.mp4       # H.264 video (libx264, crf=23, yuv420p)
      ├── left_camera-images-rgb.mp4
      ├── right_camera-images-rgb.mp4
      ├── metadata.json
      └── top_camera-images-rgb_annotation.json   # (optional) voice annotations

$YAM_RAW_PATH/eval/
  {timestamp}-YAM-{station:02d}-eval/                   # evaluation rollouts
    {YYYYMMDDTHHMMSS<microseconds>}/                    # same episode format
```

**CAP learn_skill output** (additional files compared to teleoperation):
```
      ├── is-human-action.npy             # (T,) float32 — 1.0 if human takeover, else 0.0
      └── reward.npy                      # (T,) float32 — placeholder reward signal
```

### 13.6 Metadata Schema (`metadata.json`)

**File:** `record_episode_wrapper.py:366-401`

```json
{
  "task_name": "string",
  "motion": "string (same as task_name)",
  "motion_object": "string (same as task_name)",
  "env_loop_frequency": 30,
  "duration": 10.5,
  "station_metadata": {
    "arm_type": "yam",
    "world_frame": "left_arm",
    "extrinsics": {
      "right_arm_extrinsic": {
        "position": [0.0, -0.61, 0.0],
        "rotation": [1.0, 0.0, 0.0, 0.0]
      }
    }
  },
  "attributes": {
    "manipulation_surface": "white",
    "object": "object"
  },
  "camera_info": {
    "top_camera": {
      "camera_type": "RealSenseCamera",
      "width": 640,
      "height": 480,
      "polling_fps": 60,
      "name": "top_camera"
    }
  },
  "operator": "username",
  "hostname": "machine-name",
  "policy_config": {}
}
```

### 13.7 Data Replay

**See:** [DATA_STUDIO.md](DATA_STUDIO.md) for the full replay system (overlay viz, web UI, control loop).

**`LerobotReplayPolicy`** (`experimental/lerobot_replay_policy.py:28`):
- Auto-detects input format: raw folder (NPY), Parquet, single NPY, NPZ archive
- Supports control modes: `joint_position` (14D), `delta_joint_position` (14D), `cartesian_position` (16D), `delta_ee_pose` (16D), `umi_ee_pose` (16/20D)
- Returns zero action at episode end (prevents drift in delta modes)
- Hot-reloadable via `update_replay_config()` — used by overlay viz for browsing episodes

### 13.8 Timing Instrumentation

**See:** [debug_shit_data_collection_infra.md](debug_shit_data_collection_infra.md) for the full investigation.

**`TimingJsonlLogger`** (`timing_jsonl.py:1-114`):
- Persistent JSONL logging to `/tmp/yam_timing/` (configurable)
- Each record: `{event, count, unix_time_ns, monotonic_ns, samples_ms}`
- Used across the stack: teleop policy, env, follower server, controller, main loop

**Instrumentation points:**
| Component | File | Metrics |
|-----------|------|---------|
| Main loop | `tools/vision/run_data_collection.py:175-189` | `policy_call_ms`, `env_step_ms`, `post_step_ui_ms`, `loop_total_ms` |
| Teleop policy | `teleop_policy.py:370-407` | `{side}_leader_get_info_ms`, `policy_get_action_ms` |
| Real env | `robot/yam/yam_real_env.py:368-408` | `env_get_obs_ms`, `{side}_follower_get_obs_ms`, `obs_component_span_ms` |
| Real env step | `robot/yam/yam_real_env.py:224-233` | `env_pacing_ms`, `env_pacing_overshoot_ms` |

**Known timing issue:**
- `timestamp.npy` records wall-clock time *after* `env.step()` returns, not a scheduled monotonic clock
- Host-side RPC jitter, scheduler jitter, and sequential observation assembly cause non-uniform timestamps
- Different machines exhibit different timing profiles (see cross-host benchmark in debug doc)
- Recommended fix: use `t0 + k * control_period` for canonical timestamps; keep measured latency as separate diagnostics

### 13.9 Camera Pipeline

**File:** `robot/yam/yam_real_env.py:40-98` — `NonBlockingCamera` class

- Each camera runs a background `_camera_worker` thread that continuously calls `camera.read()`
- `get_image()` returns a copy of the latest frame (non-blocking)
- Camera resolution: 640x480 at 60 fps polling rate
- Camera types: RealSense (configured via `robot/camera_factory.py`)
- Three cameras: top, left, right
- Camera timestamps in obs are host-side assembly times, not sensor-native capture times

### 13.10 IPC and Communication

| Component | Protocol | Port | File |
|-----------|----------|------|------|
| Left follower arm | Portal RPC | `LEFT_FOLLOWER_PORT` | `robot/yam/arm_server.py` |
| Right follower arm | Portal RPC | `RIGHT_FOLLOWER_PORT` | `robot/yam/arm_server.py` |
| Left leader arm (YAM) | Portal RPC | `LEFT_LEADER_PORT` | `robot/yam/arm_server.py` |
| Right leader arm (YAM) | Portal RPC | `RIGHT_LEADER_PORT` | `robot/yam/arm_server.py` |
| Left leader arm (Fello) | Portal RPC | from `fello_config.yaml` | `robot/fello/fello_server.py` |
| Right leader arm (Fello) | Portal RPC | from `fello_config.yaml` | `robot/fello/fello_server.py` |
| Overlay Viz Web UI | HTTP | 8888 | `third_party/overlay_viz/app.py` |
| Replay Control Loop | Portal IPC | 8009 | `experimental/yam_control_loop.py` |
| CAP Server | Portal RPC | 8300 | `cap/server/cap_server.py` |

### 13.11 Data Format vs External Systems

The current YAM episode format (NPY + MP4 + JSON) is a custom format optimized for simplicity and low-latency recording. It does **not** natively match any of the surveyed external formats (LeRobot, RLDS, Robo-DM).

**Conversion considerations:**
- To **LeRobot v3**: Would require converting NPY arrays to Parquet shards, MP4 videos are already compatible, metadata needs schema mapping to `info.json` + `tasks.jsonl`
- To **RLDS/TFRecord**: Would require serialization to TFRecord episode format
- The `LerobotReplayPolicy` already reads Parquet format, indicating some LeRobot integration exists for consumption but not for recording

**Current format strengths:**
- Minimal dependencies (numpy + OpenCV + ffmpeg)
- Fast append-only recording (no Parquet overhead during collection)
- Human-readable directory structure
- Compatible with the overlay viz scanner (see [DATA_STUDIO.md](DATA_STUDIO.md))

**Current format weaknesses (for training):**
- No columnar random access (each obs key is a separate file)
- No built-in episode indexing or sharding
- Video decoding required for image observations during training
- No normalization statistics computed at recording time

### 13.12 Tmux Launch Architecture

**File:** `launch.py:36-187`

The `TmuxSession` class (`launch.py:36-103`) orchestrates multi-process startup:

```
Session: "robots"
  ├── follow_l   — uv run robot/yam/arm_server.py --mode follower --side left
  ├── follow_r   — uv run robot/yam/arm_server.py --mode follower --side right
  ├── leader_l   — uv run robot/fello/fello_server.py --side left  (or YAM leader)
  ├── leader_r   — uv run robot/fello/fello_server.py --side right (or YAM leader)
  └── motor_temps — uv run robot/monitor_motor_temps.py

Session: "cameras"
  ├── top
  ├── left
  └── right

Session: "main"
  └── main       — uv run python tools/vision/run_data_collection.py [flags]
```

**Available modes** (`launch.py:107`):
- `dev` — teleop only (no recording)
- `data_collection` — full data collection with recording
- `a5_data_collection` — task-list-driven data collection
- `evaluation` — placeholder for evaluation scripts

---

*Last updated: 2026-04-08*
