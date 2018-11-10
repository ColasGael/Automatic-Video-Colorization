# CS230-Final-Project

### Converting videos

1. Create the data directories
```
mkdir data; mkdir data/raw; mkdir data/converted;
```
2. Place videos inside 'data/raw' directory
3. Run the conversion script

For all videos inside 'data/raw' directory
```
python3 converter.py
```

For one specific video 'filename'
```
python3 converter.py --inputname filename
```

To convert all videos in the data/raw folder to a consistent fps and resolution:
```
python3 converter.py --fps 30 --out_dim 640 360
```

#### Moments in Time (Mini) Dataset
Download and unzip the dataset
```
wget http://data.csail.mit.edu/soundnet/actions3/split1/Moments_in_Time_Mini.zip
unzip Moments_in_Time_Mini.zip -d data/.
```
Pre-process the dataset
```
./convert_moment_dataset.sh
```

### Requirements

### Dependencies

You can install Python dependencies using `pip install -r requirements.txt`
