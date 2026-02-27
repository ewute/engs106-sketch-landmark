# Data Directory

## Setup

1. Download the **CUHK Face Sketch Database (CUFS)** from:
   - http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html

2. Place the original images under `raw/`, organized by source:
   ```
   raw/
   ├── CUHK_student/
   │   ├── photos/
   │   └── sketches/
   ├── AR/
   │   ├── photos/
   │   └── sketches/
   └── XM2VTS/
       ├── photos/
       └── sketches/
   ```

3. Run preprocessing to generate contents of `processed/` and `splits/`.

## Notes

- This directory is **not tracked by git** (see `.gitignore`).
- Ensure you comply with the licensing terms of each sub-database.
