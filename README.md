# GraphGPS_DGL

This is the repo of GraphGPS under DGL implementation.

## Run
``` bash
# Origin model
python train.py --config config/ogbg-molhiv-GPS+RWSE.yaml
python train.py --config config/zinc-GPS+RWSE.yaml

# Use sparse multi-head attention
python train.py --config config/ogbg-molhiv-GPS+RWSE+SPTrans.yaml

# not calculate PE
python train.py --test True --config config/ogbg-molhiv-GPS+RWSE.yaml

#########################
# run with log file
model=ogbg-molhiv-GPS+RWSE
name=${model}_$(date +%H_%M_%S)
python -u train.py --config config/${model}.yaml 2>&1 | tee log/${name}.log 
```