xcrun -sdk macosx metal -c backend/gpu_backend.metal -o tmp/gpu_backend.air
xcrun -sdk macosx metallib tmp/gpu_backend.air -o tmp/gpu_backend.metallib