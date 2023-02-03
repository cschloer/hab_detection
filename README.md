# hab_detection


`rsync -v -r conrad@galilei.cv.tu-berlin.de:/shared/datasets/hab/* .`

`cat log.txt | grep -A 3 "train model performance" | sed -n "s/^.*correct: \([[:digit:]]\)/\1/p"`
