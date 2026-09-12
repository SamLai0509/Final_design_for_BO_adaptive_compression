#!/bin/bash
echo "=== hostname: $(hostname)"
echo "=== GPU:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv
echo; echo "=== topo (GPU 间 NV#=NVLink, PIX/PXB/NODE/SYS=PCIe/QPI):"
nvidia-smi topo -m
echo; echo "=== nvlink -s:"
nvidia-smi nvlink -s 2>&1 | head -12
echo; echo "=== CPU:"; lscpu | grep -E "Model name|^Socket|^Core|^CPU\(s\)" | head -4
echo "=== RAM:"; free -h | sed -n '2p'
