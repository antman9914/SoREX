# SoREX: Towards Self-Explainable Social Recommendation with Relevant Ego-Path Extraction

This is the code for our paper submitted to TKDE.

## Datasets

We have put our preprocessed datasets in `data/` folder. Each preprocessed data file is composed of 6 elements, i.e. train/val/test sets, social links and pre-computed social influence values for train and test sets respectively. If you want to use your own datasets, please transform them into the required data format. 

## Model Training

You can modify the hyperparameters in `run.sh` and run the shell script to train and test our SoREX. The recommended settings for Yelp and Flickr is:

``
soc_mp_layer=2  
co_mp_layer=1   
num_walk=100    
num_hop=3       
batch_size=256
``

The recommended settings for Ciao is:

``
soc_mp_layer=2  
co_mp_layer=1   
num_walk=100    
num_hop=3       
batch_size=1024
``

The recommended settings for LastFM is:

``
soc_mp_layer=1  
co_mp_layer=2   
num_walk=400    
num_hop=3       
batch_size=512
``

For other hyperparameters not mentioned here, please follow the settings provided in `run.sh`. 