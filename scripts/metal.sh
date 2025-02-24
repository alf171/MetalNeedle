xcrun -sdk macosx metal -c src/backend/metal/backend.metal -o tmp/metal_backend.air
xcrun -sdk macosx metallib tmp/metal_backend.air -o tmp/metal_backend.metallib