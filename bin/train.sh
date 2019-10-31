set -xe
fold=0
conf=./conf/${1}.py

python -m src.cnn.main train ${conf} --fold ${fold} --gpu ${2}

