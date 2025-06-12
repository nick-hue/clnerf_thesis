#!/usr/bin/env bash
set -e

# Helper to format seconds as M:SS
fmt() {
  local T=$1
  printf "%d:%02d" $((T/60)) $((T%60))
}

#–– 1) Database creation
start_db=$(date +%s)
colmap database_creator --database_path database.db
end_db=$(date +%s)
dur_db=$((end_db - start_db))
echo "Database created at database.db"

#–– 2) Feature extraction
start_feat=$(date +%s)
colmap feature_extractor \
  --database_path database.db \
  --image_path images/ \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 1
end_feat=$(date +%s)
dur_feat=$((end_feat - start_feat))
echo "Features extracted and stored in database.db"

#–– 3) Feature matching
start_match=$(date +%s)
colmap exhaustive_matcher \
  --database_path database.db \
  --SiftMatching.use_gpu 1
end_match=$(date +%s)
dur_match=$((end_match - start_match))
echo "Exhaustive matching done"

#–– 4) Mapping
mkdir -p sparse
start_map=$(date +%s)
colmap mapper \
  --database_path database.db \
  --image_path images/ \
  --output_path sparse \
  --Mapper.ba_global_function_tolerance=1e-6
end_map=$(date +%s)
dur_map=$((end_map - start_map))
echo "Mapping done"

#–– Print all timings at the end
echo
echo "===== TIMINGS ====="
echo "1) Database creation : $(fmt $dur_db) (MM:SS)"
echo "2) Feature extraction: $(fmt $dur_feat) (MM:SS)"
echo "3) Feature matching   : $(fmt $dur_match) (MM:SS)"
echo "4) Mapping           : $(fmt $dur_map) (MM:SS)"
echo "===================="
