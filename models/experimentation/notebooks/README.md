# 3D Geometry Learning Workspace

The Open3D exercises use a separate Python 3.12 environment because Open3D
0.19.0 does not provide a wheel for SquatSpot's Python 3.13 environment.
Keeping the environments separate avoids changing the application runtime.

The environment is located at `C:\tmp\sso3d` to avoid Windows path-length
errors in Open3D's visualization dependencies.

From the SquatSpot repository root, start Lesson 1 with:

```powershell
C:\tmp\sso3d\Scripts\python.exe models\experimentation\notebooks\open3d_learn.py
```

The script loads MM-Fit session `w00`, selects frame 4270, removes MM-Fit's
timestamp row, prints coordinate evidence, and displays the remaining 17 joints
as an Open3D point cloud. Work through the `TODO` checkpoints in the script
before adding skeleton connections.
