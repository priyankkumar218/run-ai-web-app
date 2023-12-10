# Download and save train images in parallel
wget http://images.cocodataset.org/zips/train2017.zip -O train2017.zip && unzip -q train2017.zip &

# Download and save captions annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O captions_annotations_trainval2017.zip && unzip -j captions_annotations_trainval2017.zip -d annotations &

# Download and save validation images in parallel
wget http://images.cocodataset.org/zips/val2017.zip -O val2017.zip && unzip -q val2017.zip

# remove zip files
rm -rf *.zip