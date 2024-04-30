CHECKPOINT_DIR='./ckpt' 
LOG_DIR='./log'
dataset='lastfm'       
if [ ! -d $CHECKPOINT_DIR ]
then
    mkdir $CHECKPOINT_DIR
fi
if [ ! -d $LOG_DIR ]
then
    mkdir $LOG_DIR
fi

mode='train'
# mode='test'   # Uncomment this line if you test model, and don't forget to specify the time_str param

gpu_id=1
soc_mp_layer=1
co_mp_layer=2
K=6
num_walk=500
num_hop=3
input_dim=64
batch_size=512
test_bs=64
lr=0.001
weight_decay=0.001
neg_sample_num=8
eval_per_n=1000  
epoch_num=50
early_stop=5
time_str=""

note="design-alike-wo-ib-w-soc"

python -u main.py \
    --dataset=$dataset \
    --mode=$mode \
    --note=$note \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --log_dir=$LOG_DIR \
    --gpu_id=$gpu_id \
    --soc_mp_layer=$soc_mp_layer \
    --co_mp_layer=$co_mp_layer \
    --K=$K \
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
    