echo "[1] Copying local files to /opt/conda/lib/python3.10/site-packages/mmdet3d and mmengine "
cp /workspace/replace_mmdetection_files/loops.py /opt/conda/lib/python3.10/site-packages/mmengine/runner/
cp /workspace/replace_mmdetection_files/base_model.py /opt/conda/lib/python3.10/site-packages/mmengine/model/base_model/
cp /workspace/replace_mmdetection_files/transforms_3d.py /opt/conda/lib/python3.10/site-packages/mmdet3d/datasets/transforms/

echo "[2] Moving the segmentor to the right place"
cp /segmentator/ /workspace/segmentator -r

ln -s /segmentator/csrc/build/ /workspace/segmentator/csrc/buil