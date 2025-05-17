python /media/agent/maab/tools/scripts/process_autokaggle.py \
  /media/agent/maab/datasets/airbnb_melbourne \
  --train train.pq \
  --test inference.pq
  
python /media/agent/maab/tools/scripts/process_autokaggle.py \
  /media/agent/maab/datasets/airlines \
  --train train.csv \
  --test test.csv

python /media/agent/maab/tools/scripts/process_autokaggle.py \
  /media/agent/maab/datasets/cd18 \
  --train train.csv \
  --test inference.csv

python /media/agent/maab/tools/scripts/process_autokaggle.py \
  /media/agent/maab/datasets/covertype \
  --train train.csv \
  --test test.csv

python /media/agent/maab/tools/scripts/process_autokaggle.py \
  /media/agent/maab/datasets/gnad10 \
  --train train.csv \
  --test inference.csv

python /media/agent/maab/tools/scripts/process_autokaggle.py \
  /media/agent/maab/datasets/ham10000 \
  --train ham10000_train_annotations.csv \
  --test inference.csv

python /media/agent/maab/tools/scripts/process_autokaggle.py \
  /media/agent/maab/datasets/hateful_meme \
  --train train.csv \
  --test inference.csv

python /media/agent/maab/tools/scripts/process_autokaggle.py \
  /media/agent/maab/datasets/kick_starter_funding \
  --train train.csv \
  --test inference.csv

python /media/agent/maab/tools/scripts/process_autokaggle.py \
  /media/agent/maab/datasets/petfinder \
  --train train.csv \
  --test inference.csv

python /media/agent/maab/tools/scripts/process_autokaggle.py \
  /media/agent/maab/datasets/women_clothing_review \
  --train train.pq \
  --test inference.pq

python /media/agent/maab/tools/scripts/process_autokaggle.py \
  /media/agent/maab/datasets/yolanda \
  --train train.csv \
  --test test.csv
  
# For airbnb_melbourne
ln -s /media/agent/maab/datasets/airbnb_melbourne/autokaggle /media/agent/AutoKaggle/multi_agents/competition/airbnb_melbourne

# For airlines
ln -s /media/agent/maab/datasets/airlines/autokaggle /media/agent/AutoKaggle/multi_agents/competition/airlines

# For cd18
ln -s /media/agent/maab/datasets/cd18/autokaggle /media/agent/AutoKaggle/multi_agents/competition/cd18

# For covertype
ln -s /media/agent/maab/datasets/covertype/autokaggle /media/agent/AutoKaggle/multi_agents/competition/covertype

# For gnad10
ln -s /media/agent/maab/datasets/gnad10/autokaggle /media/agent/AutoKaggle/multi_agents/competition/gnad10

# For ham10000
ln -s /media/agent/maab/datasets/ham10000/autokaggle /media/agent/AutoKaggle/multi_agents/competition/ham10000

# For hateful_meme
ln -s /media/agent/maab/datasets/hateful_meme/autokaggle /media/agent/AutoKaggle/multi_agents/competition/hateful_meme

# For kick_starter_funding
ln -s /media/agent/maab/datasets/kick_starter_funding/autokaggle /media/agent/AutoKaggle/multi_agents/competition/kick_starter_funding

# For petfinder
ln -s /media/agent/maab/datasets/petfinder/autokaggle /media/agent/AutoKaggle/multi_agents/competition/petfinder

# For women_clothing_review
ln -s /media/agent/maab/datasets/women_clothing_review/autokaggle /media/agent/AutoKaggle/multi_agents/competition/women_clothing_review

# For yolanda
ln -s /media/agent/maab/datasets/yolanda/autokaggle /media/agent/AutoKaggle/multi_agents/competition/yolanda