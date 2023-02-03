# hab_detection

`cat log.txt | grep -A 3 "train model performance" | sed -n "s/^.*correct: \([[:digit:]]\)/\1/p"`
