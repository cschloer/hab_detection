# hab_detection


`rsync -v -r conrad@galilei.cv.tu-berlin.de:/shared/datasets/hab/* .`

`cat log.txt | grep -A 3 "train model performance" | sed -n "s/^.*correct: \([[:digit:]]\)/\1/p"`

`gdown 1-2-DwVMcR9sV7kZQ2J2dHQIokXlL4iVZ`

`scp conrad@portia.cv.tu-berlin.de:/shared/datasets/hab/models/experiment12/visualize:`


docker build -t dataset .
docker run -d --env-file ./dataset_create/.env -v /shared/datasets/hab/new_data:/shared/datasets/hab/new_data -v /home/conrad:/home/conrad dataset
