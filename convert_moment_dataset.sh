if [ ! -d data/Moments_in_Time_Mini ]; then
  echo "Moments_in_Time_Mini dataset not downloaded";
  exit;
fi

mkdir -p data/Moments_processed;

for directory in $(find data/Moments_in_Time_Mini/training -type d -mindepth 1); 
do
    echo "Converting videos in directory $directory"; 
    python3 converter.py --input_dir "$directory/" --output_dir data/Moments_processed/;
done
