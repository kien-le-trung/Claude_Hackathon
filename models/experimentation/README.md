# Squat-form experimentation

This is the canonical local workspace for landmark-based squat-form models. The
source images are the *Squat Classification Dataset* (Zenodo record 17558630,
CC BY 4.0). The supplied split contains sequential frames and may produce
optimistic metrics; every promoted artifact records that limitation.

All commands must use the repository's existing virtual environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r apps\api\requirements.txt
.\venv\Scripts\python.exe -m models.experimentation.cli audit
.\venv\Scripts\python.exe -m models.experimentation.cli extract
.\venv\Scripts\python.exe -m models.experimentation.cli train
.\venv\Scripts\python.exe -m models.experimentation.cli promote --version v2
```

Inputs default to `assets/data/{train,test}`. Generated manifests and landmark
caches live in `models/experimentation/data`, while model runs live in
`models/experimentation/runs`; both directories are ignored. Approved ONNX
artifacts are copied to `models/runtime/squat_form/<version>`. The required,
lightweight v2 ONNX artifact and its metadata are retained with the application.

The stages are intentionally separate: `audit` inventories and hashes images,
`extract` caches MediaPipe results, `train` benchmarks configured feature/model
combinations, and `promote` enforces quality gates before publishing an artifact.
