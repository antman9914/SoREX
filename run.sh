CHECKPOINT_DIR='./ckpt'     # Directory path for model checkpoint
LOG_DIR='./log'             # Directory path for training logs
dataset='lastfm'            # Specify datasets from ['yelp','flickr','ciao','lastfm']
if [ ! -d $CHECKPOINT_DIR ]
then
    mkdir $CHECKPOINT_DIR
fi
if [ ! -d $LOG_DIR ]
then
    mkdir $LOG_DIR
fi

mode='train'
# mode='test'   # Uncomment this line if you test model, and don't forget to specify the time_str param during testing

gpu_id=1        # Specify GPU id. Please set the param as -1 if you'd like to run in CPU mode. 
soc_mp_layer=1  # Social tower LightGCN layer number
co_mp_layer=2   # Interaction tower LightGCN layer number
num_walk=500    # Number of sampled ego-paths
num_hop=3       # Hop number of ego-paths
input_dim=64    # Embedding dimension
batch_size=512
test_bs=64      # Batch size for testing
lr=0.001
weight_decay=0.001
neg_sample_num=8
eval_per_n=1000  
epoch_num=50
early_stop=5
time_str=""

note="SoREX-Default"     # Note for current run. This is completely irrelevant with the training/testing code

python -u main.py \
    --dataset=$dataset \
    --mode=$mode \
    --note=$note \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --log_dir=$LOG_DIR \
    --gpu_id=$gpu_id \
    --soc_mp_layer=$soc_mp_layer \
    --co_mp_layer=$co_mp_layer \
    --num_walk=$num_walk \
    --num_hop=$num_hop \
    --input_dim=$input_dim \
    --weight_decay=$weight_decay \
    --lr=$lr \
    --batch_size=$batch_size \
    --test_bs=$test_bs \
    --neg_sample_num=$neg_sample_num \
    --eval_per_n=$eval_per_n \
    --epoch_num=$epoch_num \
    --early_stop=$early_stop \
    