#! /bin/sh
export path_aux=$1

echo "Downloading semantic embeddings and pre-trained models (it will take some time)"
python3 ../download_gdrive.py 16xvgqy5FFBqxFua7I7TZMxJFEuR3mQ3b $path_aux/aux_files.zip
echo "Unzipping..."
unzip $path_aux/aux_files.zip -d $path_aux
rm -rfv $path_aux/aux_files.zip
echo "Done"
echo "Cleaning up redundant files..."
mv $path_aux/CheckPoints/Sketchy/sketch/model_best.pth $path_aux/pretrained/vgg16_sketch.pth
mv $path_aux/CheckPoints/Sketchy/image/model_best.pth $path_aux/pretrained/vgg16_photo.pth
rm -rfv $path_aux/Semantics
rm -rfv $path_aux/CheckPoints/*
echo "Pre-trained models are now ready to be used"
