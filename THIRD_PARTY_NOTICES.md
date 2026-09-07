# Third-party notices

| Component | Source | Licence | Use in this entry |
|---|---|---|---|
| Microduck MJCF model (`assets/microduck/robot_groundcontact.xml`) | pollen-robotics/microduck_rl @ 29e887e | Apache-2.0 | used unmodified, attached 5× with name prefixes; shell colours changed per duck |
| Microduck 3D meshes (`assets/microduck/assets/*.stl`) | pollen-robotics/microduck_rl | CC BY-SA-NC 4.0 (Pollen Robotics) | rendering and collision, unmodified; this entry is non-commercial |
| `alpha_walking.onnx`, `alpha_stand.onnx`, `alpha_sitstand.onnx` | huggingface.co/pollen-robotics/microduck-policies | Apache-2.0 | the learned gait / stand / sit-stand every duck runs, unmodified |
| `standup.onnx` | trained for this entry with microduck_rl's `Mjlab-StandUp-Flat-MicroDuck` task (Pollen's environment, reward and exporter; our compute) | Apache-2.0 | stand-up recovery for every duck |
| BAM actuator conversion (`duckman/bam_loader.py`), default pose and constants | microduck_rl `scripts/infer_policy.py` | Apache-2.0 | adapted (takes an `MjSpec`) |
| better-actuator-models (BAM, XL330 M6 model) | github.com/Rhoban/bam @ 62bd8ce (branch `mjlab_frictionloss`) | MIT | actuator physics |
| MuJoCo | github.com/google-deepmind/mujoco | Apache-2.0 | physics and rendering |
| onnxruntime, numpy, imageio, imageio-ffmpeg (bundles FFmpeg, LGPL), Pillow, pytest | PyPI | respective licences | inference, video, tests |

Everything under `duckman/` and `tests/` not listed above was written for this entry and is
released under Apache-2.0 (see LICENSE).
