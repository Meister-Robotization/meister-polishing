# meister-polishing

Python scripts that record human hand motion from a camera and generate robot trajectories via DMP (Dynamic Movement Primitive) processing.

## Required packages

- **opencv-python** (`cv2`) — image processing and display
- **numpy** — numerical operations
- **torch** — optional, for GPU
- **mediapipe** — hand landmark detection (code_0)
- **pyrealsense2** — RealSense camera (used in some classes in common)
- **pyorbbecsdk** — Orbbec Femto Bolt camera (code_0; vendor install may be required)
- **dt-apriltags** — AprilTag marker detection (common)
- **matplotlib** — plotting (code_1)
- **h5py**, **dm_env**, **supervision**, **Pillow** — code_1 dependencies
- **pyserial** — serial communication (common)

`common.py` imports `utils` from `/root/Projects/pyorbbecsdk/examples/`. Adjust that path or `sys.path` to match your Orbbec SDK install.

## Code overview

| File | Role |
|------|------|
| **code_0_record_human_demo_femto.py** | Tracks the hand with Femto Bolt (Orbbec) + MediaPipe and logs hand position (x,y,z) to a text file. `[r]` start recording, `[s]` stop, `[q]` quit. |
| **code_1_process_human_demo.py** | Loads the recorded demo text, applies LPF, camera-angle compensation, and axis transform, then generates a robot trajectory with DMP and saves it to a file. |
| **common.py** | Shared utilities: cameras (femtoBolt, RealSense), AprilTag detection, DMP (`MovementPrimitive`, `DMPPathGenerator`), etc. |

## How to run

### code_0 (record hand motion)

1. Set `PATH_TXT` at the top of `code_0_record_human_demo_femto.py` to your desired output text path.
2. Connect the Femto Bolt camera and run:

```bash
cd code
python code_0_record_human_demo_femto.py
```

3. In the window: `r` → record, `s` → stop, `q` or ESC → quit. Recorded data is written to `PATH_TXT`.

### code_1 (process demo and generate trajectory)

1. In `code_1_process_human_demo.py`, set:
   - `PATH_TXT`: path to the human-demo text file from code_0
   - `OUT_TXT`: path where the DMP trajectory will be saved
   - `start`, `end`: robot trajectory start/end coordinates (x, y, z [mm])
2. Run:

```bash
cd code
python code_1_process_human_demo.py
```

The DMP trajectory is saved to `OUT_TXT`.
