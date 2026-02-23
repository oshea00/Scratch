# Stacked DGX Sparks


[stacked-sparks](https://build.nvidia.com/spark/nccl/stacked-sparks)
[system requirements](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html#system-requirements)


### My nodes
169.254.103.215/16 - spark2 enp1s0f1np1
169.254.113.4/16 - spark1 enp1s0f1np1

```bash
# Set network interface environment variables (use your Up interface from the previous step)
# Set environment variables
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
export PORT_NAME=enp1s0f1np1
export UCX_NET_DEVICES=$PORT_NAME
export NCCL_SOCKET_IFNAME=$PORT_NAME
export OMPI_MCA_btl_tcp_if_include=$PORT_NAME
export DEVICE_1_IP=169.254.113.4
export DEVICE_2_IP=169.254.103.215

# Run the all_gather performance test across both nodes (replace the IP addresses with the ones you found in the previous step)
mpirun -np 2 -H $DEVICE_1_IP:1,$DEVICE_2_IP:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

## Various tools
* ibdev2netdev
* cat /etc/dgx-release
* nvidia-smi topo -m
* ethtool enp1s0f1np1

## A few words on cables
The DGX Spark CX-7 ports support ethernet configuration only.

Approved cables for the CX-7 ports are:
** Amphenol: NJAAKK-N911 (QSFP to QSFP112, 32AWG, 400mm, LSZH), NJAAKK0006 is the 0.5m version of this cable
Luxshare: LMTQF022-SD-R (QSFP112 400G DAC Cable, 400mm, 30AWG)

### VLLM on two sparks

```bash
export LATEST_VLLM_VERSION=26.01-py3
docker pull nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION}

### Single spark test (recommend doing on both sparks first)

Launch container:
docker run -it --gpus all -p 8000:8000 \
nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
vllm serve "Qwen/Qwen2.5-Math-1.5B-Instruct"

curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [{"role": "user", "content": "12*17"}],
    "max_tokens": 500
}'
```

### Run on two sparks
```bash
# Download on both nodes
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/online_serving/run_cluster.sh
chmod +x run_cluster.sh
```
### start Ray head

```bash
# On Node 1, start head node

# Get the IP address of the high-speed interface
# Use the interface that shows "(Up)" from ibdev2netdev (enp1s0f0np0 or enp1s0f1np1)
export MN_IF_NAME=enp1s0f1np1
export LATEST_VLLM_VERSION=26.01-py3
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export VLLM_IMAGE=nvcr.io/nvidia/vllm:$LATEST_VLLM_VERSION

echo "Using interface $MN_IF_NAME with IP $VLLM_HOST_IP"

bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$VLLM_HOST_IP
```


### start Ray workerecho

```bash
# On Node 2, join as worker
# Set the interface name (same as Node 1)
export MN_IF_NAME=enp1s0f1np1
export LATEST_VLLM_VERSION=26.01-py3
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export VLLM_IMAGE=nvcr.io/nvidia/vllm:$LATEST_VLLM_VERSION

# IMPORTANT: Set HEAD_NODE_IP to Node 1's IP address
# You must get this value from Node 1 (run: echo $VLLM_HOST_IP on Node 1)
export HEAD_NODE_IP=169.254.113.4

echo "Worker IP: $VLLM_HOST_IP, connecting to head node at: $HEAD_NODE_IP"

bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$HEAD_NODE_IP
```

```bash
# Check cluster status (from head node)
# On Node 1 (head node)
# Find the vLLM container name (it will be node-<random_number>)
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
echo "Found container: $VLLM_CONTAINER"
docker exec $VLLM_CONTAINER ray status
```

```bash
# Download Llama 3.3 70B (on head and tail)
# From within the same container where `ray status` ran, run the following
hf auth login
hf download meta-llama/Llama-3.3-70B-Instruct
```

### Launch
```bash
# On Node 1, enter container and start server
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 --max_model_len 2048'
```
### Test from Node 1 or external client

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "prompt": "Write a haiku about a GPU",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

