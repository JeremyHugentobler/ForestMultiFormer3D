wget -O weights.zip https://zenodo.org/records/16742708/preview/clean_forestformer.zip?include_deleted=0#tree_item0
mkdir work_dirs
mv weights.zip work_dirs/
unzip work_dirs/weights.zip
rm work_dirs/weights.zip
