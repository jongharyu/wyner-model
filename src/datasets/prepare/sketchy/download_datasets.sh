#! /bin/sh
path_to_dataset=$1

# download the Sketchy dataset
echo "[1] Downloading the Sketchy dataset (it will take some time)"
python3 ../download_gdrive.py 0B7ISyeE8QtDdTjE1MG9Gcy1kSkE $path_to_dataset/Sketchy.7z
echo "Unzipping Sketchy.7z..."
7z x $path_to_dataset/Sketchy.7z o $path_to_dataset > $path_to_dataset/garbage.txt
rm $path_to_dataset/garbage.txt
rm $path_to_dataset/Sketchy.7z
rm $path_to_dataset/README.txt
mv $path_to_dataset/256x256 $path_to_dataset/Sketchy
echo "Done"

echo "[2] Downloading the extended photos of Sketchy dataset (it will take some time)"
python3 src/download_gdrive.py 0B2U-hnwRkpRrdGZKTzkwbkEwVkk $path_to_dataset/Sketchy/extended_photo.zip
echo "Unzipping extended_photo.zip..."
unzip -qq $path_to_dataset/Sketchy/extended_photo.zip -d $path_to_dataset/Sketchy
rm $path_to_dataset/Sketchy/extended_photo.zip
mv $path_to_dataset/Sketchy/EXTEND_image_sketchy $path_to_dataset/Sketchy/extended_photo
echo "Done"

echo "[3] Cleaning and rearranging dataset folders..."
echo "  [3-1] Removing redundant sketch images other than tx_000000000000..."
rm -r $path_to_dataset/Sketchy/sketch/tx_000000000010
rm -r $path_to_dataset/Sketchy/sketch/tx_000000000110
rm -r $path_to_dataset/Sketchy/sketch/tx_000000001010
rm -r $path_to_dataset/Sketchy/sketch/tx_000000001110
rm -r $path_to_dataset/Sketchy/sketch/tx_000100000000
echo "  [3-2] Removing redundant photo images other than tx_000000000000..."
rm -r $path_to_dataset/Sketchy/photo/tx_000100000000
echo "  [3-3] Renaming sketch folders (hot_air_balloon, jack_o_lantern)..."
mv $path_to_dataset/Sketchy/sketch/tx_000000000000/hot-air_balloon $path_to_dataset/Sketchy/sketch/tx_000000000000/hot_air_balloon
mv $path_to_dataset/Sketchy/sketch/tx_000000000000/jack-o-lantern $path_to_dataset/Sketchy/sketch/tx_000000000000/jack_o_lantern
echo "  [3-4] Renaming photo folders (hot_air_balloon, jack_o_lantern)..."
mv $path_to_dataset/Sketchy/photo/tx_000000000000/hot-air_balloon $path_to_dataset/Sketchy/photo/tx_000000000000/hot_air_balloon
mv $path_to_dataset/Sketchy/photo/tx_000000000000/jack-o-lantern $path_to_dataset/Sketchy/photo/tx_000000000000/jack_o_lantern
echo "  [3-5] Renaming extended photo folders (hot_air_balloon, jack_o_lantern)..."
mv $path_to_dataset/Sketchy/extended_photo/hot-air_balloon $path_to_dataset/Sketchy/extended_photo/hot_air_balloon
mv $path_to_dataset/Sketchy/extended_photo/jack-o-lantern $path_to_dataset/Sketchy/extended_photo/jack_o_lantern
echo "Done"
echo "Sketchy dataset is now ready to be used"
