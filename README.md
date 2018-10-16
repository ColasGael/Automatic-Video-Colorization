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

## Requirements

### Dependencies

You can install Python dependencies using `pip install -r requirements.txt`
