# MLKV: Efficiently Scaling up Large Embedding Model Training with Disk-based Key-Value Storage
MLKV is a research storage framework that tackles scalability issues of embedding model training, including memory limitations, data staleness, and cache inefficiencies.
MLKV augments key-value storage to provide easy-to-use and non-intrusive interfaces for various machine-learning tasks.

## Applications
* [MLKV-DLRM](https://github.com/ml-kv/mlkv-dlrm): deep learning-based recommendations
* [MLKV-GNN](https://github.com/ml-kv/mlkv-gnn): graph neural networks
* [MLKV-KG](https://github.com/ml-kv/mlkv-kg): knowledge graphs
* [MLKV-PYTHON](https://github.com/ml-kv/mlkv-gnn/blob/dgl-mlkv/python/dgl/storages/mlkv.py): python wrapper

## build
```
sudo apt-get install make cmake clang
sudo apt-get install uuid-dev libaio-dev libtbb-dev
git clone -b main https://github.com/ml-kv/mlkv-cpp mlkv-cpp
mkdir mlkv-cpp/cc/build
cd mlkv-cpp/cc/build
CC=clang CXX=clang++ cmake ../ -DCMAKE_BUILD_TYPE=[Debug/Release/RelWithDebInfo]
make -j8
```
## benchmark
```
./process_ycsb load_250M_raw.dat load_250M.dat
./process_ycsb run_uniform_250M_1000M_raw.dat run_uniform_250M_1000M.dat
./benchmark 0 8 load_250M.dat run_uniform_250M_1000M.dat
```
`benchmark` that runs MLKV

`variable_length_benchmark` that runs MLKV under varying value sizes

`recover_benchmark` that checkpoints and recovers MLKV

# [FASTER](https://github.com/microsoft/FASTER)
## build
```
sudo apt-get install make cmake clang
sudo apt-get install uuid-dev libaio-dev libtbb-dev
git clone -b FASTER-v2.0.19 https://github.com/ml-kv/mlkv-cpp/ FASTER-v2.0.19
mkdir FASTER-v2.0.19/cc/build
cd FASTER-v2.0.19/cc/build
CC=clang CXX=clang++ cmake ../ -DCMAKE_BUILD_TYPE=[Debug/Release/RelWithDebInfo]
make -j8
```
## benchmark
```
./process_ycsb load_250M_raw.dat load_250M.dat
./process_ycsb run_uniform_250M_1000M_raw.dat run_uniform_250M_1000M.dat
./benchmark 0 8 load_250M.dat run_uniform_250M_1000M.dat
```

# [YCSB](https://github.com/brianfrankcooper/YCSB)
```
sudo apt install python2.7 default-jre
curl -O --location https://github.com/brianfrankcooper/YCSB/releases/download/0.17.0/ycsb-0.17.0.tar.gz
tar xfvz ycsb-0.17.0.tar.gz
cd ycsb-0.17.0
./bin/ycsb load basic -P workloads/workloada -p recordcount=250000000 -s > load_250M_raw.dat
./bin/ycsb.sh run basic -P workloads/workloada -p recordcount=250000000 -p operationcount=1000000000 -p requestdistribution=uniform -s > run_uniform_250M_1000M_raw.dat
```

# [FIO](https://github.com/axboe/fio)
```
sudo apt-get install fio
fio --randrepeat=1 --ioengine=libaio --direct=1 --gtod_reduce=1 --name=fiotest --filename=testfio --bs=4k --iodepth=64 --size=8G --readwrite=randrw --rwmixread=50
```
