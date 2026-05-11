# AutoDL 上复现 HybriMoE 编译安装

本文记录一次已经跑通的 AutoDL 环境配置，用于从源码编译安装 HybriMoE 里的 `ktransformers`，并验证 native extensions 可以正常导入。重点是固定 `ktransformers` 源码版本和 `llama.cpp` 的兼容版本，避免跟随当前 submodule gitlink 进入不兼容的新版 ggml API。

## 0. 已验证环境

本次成功环境：

```text
镜像: cuda12.4-cudnn-devel-ubuntu22.04-py312-torch2.5.1
GPU: NVIDIA GeForce RTX 4090D, 24GB
CUDA: 12.4, nvcc V12.4.131
系统 gcc/g++: 11.4.0
Python: conda Python 3.11
Torch: 2.5.1+cu124
ktransformers: 0.2.3post2
HybriMoE commit: 508a85b9a43dff37124370d8816e411e74827803
pybind11 submodule: c125cc789ca1be95bcd10479c568d5d260c743ad
llama.cpp build commit: e112b610a1a75cb7fa8351e1a933e2e7a755a5ce
```

注意：父仓库当前记录的 `third_party/llama.cpp` gitlink 是 `2189fd3b6327a1d17893694125da8edcf74a6468`，但这个版本对应 2025 年新版 ggml 目录和 API，和 HybriMoE 当前 C++ wrapper 不兼容。本次成功做法是先按 submodule 初始化，再把 `third_party/llama.cpp` 手动回滚到 `e112b610...`。

## 1. 工作目录和环境变量

把 conda 环境、HF cache、日志和模型都放到数据盘：

```bash
export WORK=/root/autodl-tmp/benchmark
mkdir -p "$WORK"/{conda_envs,conda_pkgs,hf,logs,models,src}

export CONDA_PKGS_DIRS=$WORK/conda_pkgs
export HF_HOME=$WORK/hf
export TRANSFORMERS_CACHE=$WORK/hf
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
```

创建 Python 3.11 conda 环境：

```bash
conda create -y -p "$WORK/conda_envs/hybrimoe-py311" python=3.11
conda activate "$WORK/conda_envs/hybrimoe-py311"

python -m pip install -U pip setuptools wheel packaging
```

不需要 `apt-get`。如果系统里已有 `/usr/bin/gcc` 和 `/usr/bin/g++`，优先使用系统编译器即可。本次成功环境使用的是 Ubuntu 22.04 自带 gcc/g++ 11.4.0。

安装基础构建工具：

```bash
conda install -y -c conda-forge "cmake>=3.26,<4" ninja git git-lfs make
```

安装 CUDA 12.4 对应的 PyTorch：

```bash
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

## 2. 获取源码和 submodule

如果 GitHub HTTPS 不稳定，建议全程使用 SSH。先确认 SSH 可用：

```bash
ssh -T -o StrictHostKeyChecking=accept-new git@github.com
```

如果 22 端口不稳定，可以使用 GitHub SSH over 443：

```bash
mkdir -p ~/.ssh
cat >> ~/.ssh/config <<'EOF'
Host github.com
  HostName ssh.github.com
  User git
  Port 443
  StrictHostKeyChecking accept-new
EOF
chmod 600 ~/.ssh/config

ssh -T git@github.com
```

克隆 HybriMoE 并固定父仓库版本：

```bash
cd "$WORK/src"
git clone git@github.com:PKU-SEC-Lab/HybriMoE.git
cd HybriMoE
git checkout 508a85b9a43dff37124370d8816e411e74827803
```

初始化 submodule。这里允许先进入父仓库记录的版本，因为后面会单独回滚 `llama.cpp`：

```bash
git submodule init
git submodule set-url third_party/llama.cpp git@github.com:ggerganov/llama.cpp.git
git submodule set-url third_party/pybind11 git@github.com:pybind/pybind11.git
git submodule sync --recursive
git submodule update --init --recursive --jobs 4

git -C third_party/llama.cpp rev-parse HEAD
git -C third_party/pybind11 rev-parse HEAD
```

此时通常会看到：

```text
2189fd3b6327a1d17893694125da8edcf74a6468
c125cc789ca1be95bcd10479c568d5d260c743ad
```

## 3. 回滚 llama.cpp 到兼容版本

HybriMoE 当前源码中有大量旧 ggml API：

```text
llama.cpp/ggml.h
llama.cpp/ggml-common.h
llama.cpp/ggml-impl.h
llama.cpp/ggml-quants.h
GGML_TASK_TYPE_COMPUTE
ggml_internal_get_type_traits(...)
```

这些在 `2189fd3...` 的新版 ggml 中已经迁移或删除。因此必须把 `third_party/llama.cpp` 回滚到 `e112b610...`：

```bash
cd "$WORK/src/HybriMoE"

git -C third_party/llama.cpp remote set-url origin git@github.com:ggerganov/llama.cpp.git
git -C third_party/llama.cpp fetch --depth 1 origin e112b610a1a75cb7fa8351e1a933e2e7a755a5ce
git -C third_party/llama.cpp checkout --detach e112b610a1a75cb7fa8351e1a933e2e7a755a5ce
```

如果之前尝试过给新版 `llama.cpp` 加转发头，或者 checkout 后出现 `?? ggml/`，先恢复旧版本 tracked 头文件，再把新版残留目录移走：

```bash
git -C third_party/llama.cpp restore -- ggml.h ggml-common.h ggml-impl.h ggml-quants.h

BACKUP=/root/autodl-tmp/benchmark/logs/llama_new_layout_backup_$(date +%F_%H%M%S)
mkdir -p "$BACKUP"
if [ -d third_party/llama.cpp/ggml ] && [ -z "$(git -C third_party/llama.cpp ls-files ggml)" ]; then
  mv third_party/llama.cpp/ggml "$BACKUP/ggml"
fi
```

验证旧头文件和旧 API 确实存在：

```bash
git -C third_party/llama.cpp rev-parse HEAD
git -C third_party/llama.cpp status --short

test -f third_party/llama.cpp/ggml.h
test -f third_party/llama.cpp/ggml-common.h
test -f third_party/llama.cpp/ggml-impl.h
test -f third_party/llama.cpp/ggml-quants.h

grep -n "GGML_TASK_TYPE_COMPUTE" third_party/llama.cpp/ggml.h
grep -n "ggml_internal_get_type_traits" third_party/llama.cpp/ggml.h
```

成功时应看到：

```text
e112b610a1a75cb7fa8351e1a933e2e7a755a5ce
683:        GGML_TASK_TYPE_COMPUTE,
2454:    GGML_API ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type type);
```

从这一步以后，不要再执行：

```bash
git submodule update --init --recursive
```

否则 `third_party/llama.cpp` 会被父仓库 gitlink 拉回 `2189fd3...`，再次触发不兼容问题。

## 4. 修复 fast-math 编译问题

旧版和新版 ggml 都会拒绝 `-ffast-math`，典型错误是：

```text
#error "some routines in ggml.c require non-finite math arithmetics -- pass -fno-finite-math-only to the compiler to fix"
```

把 HybriMoE 的 CMake 编译选项改成 `-fno-finite-math-only`：

```bash
cd "$WORK/src/HybriMoE"

python - <<'PY'
from pathlib import Path
p = Path("ktransformers/ktransformers_ext/CMakeLists.txt")
s = p.read_text()
old = 'set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")'
new = 'set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-finite-math-only")'
if old in s:
    p.write_text(s.replace(old, new))
elif new in s:
    print("already patched")
else:
    raise SystemExit("CMAKE_CXX_FLAGS line not found; inspect CMakeLists.txt")
PY

grep -n "CMAKE_CXX_FLAGS" ktransformers/ktransformers_ext/CMakeLists.txt
```

## 5. 安装 Python 依赖

```bash
cd "$WORK/src/HybriMoE"

python -m pip install -r requirements-local_chat.txt
python -m pip install \
  ninja "cmake>=3.26,<4" cpufeature pyyaml tqdm sentencepiece accelerate \
  protobuf tiktoken blobfile fire colorlog
```

## 6. 编译安装 ktransformers

清理旧 build cache：

```bash
cd "$WORK/src/HybriMoE"

rm -rf build *.egg-info \
       ktransformers/ktransformers_ext/build \
       ktransformers/ktransformers_ext/cuda/build \
       ktransformers/ktransformers_ext/cuda/dist \
       ktransformers/ktransformers_ext/cuda/*.egg-info
```

设置编译环境并安装：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export MAX_JOBS=6
export CMAKE_BUILD_PARALLEL_LEVEL=6
export CPU_INSTRUCT=NATIVE
export TORCH_CUDA_ARCH_LIST="8.9"
export KTRANSFORMERS_FORCE_BUILD=TRUE
export CMAKE_ARGS="-DLLAMA_LLAMAFILE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF"

python -m pip install . --no-build-isolation -v \
  2>&1 | tee /root/autodl-tmp/benchmark/logs/hybrimoe_build_llama_e112b610_$(date +%F_%H%M%S).log
```

成功日志应包含：

```text
Building wheel for ktransformers (pyproject.toml): finished with status 'done'
Created wheel for ktransformers: ktransformers-0.2.3.post2+torch25fancy-cp311-cp311-linux_x86_64.whl
Successfully built ktransformers
Successfully installed ... ktransformers-0.2.3.post2+torch25fancy
```

同时 wheel 内应包含：

```text
KTransformersOps.cpython-311-x86_64-linux-gnu.so
cpuinfer_ext.cpython-311-x86_64-linux-gnu.so
```

## 7. 导入验证

```bash
cd "$WORK/src/HybriMoE"

python - <<'PY'
import torch
import ktransformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("ktransformers", ktransformers.__version__)

import cpuinfer_ext
import KTransformersOps
print("native extensions ok")
PY
```

本次成功输出：

```text
torch 2.5.1+cu124 cuda 12.4
ktransformers 0.2.3post2
native extensions ok
```

看到以上输出，说明源码编译安装和 native extensions 已经跑通。

## 8. 模型运行入口

安装成功后再准备模型权重。README 中的轻量入口是 DeepSeek-V2-Lite-Chat：

```bash
mkdir -p "$WORK/models/DeepSeek-V2-Lite-Chat-GGUF"
cd "$WORK/models/DeepSeek-V2-Lite-Chat-GGUF"

wget https://huggingface.co/mzwing/DeepSeek-V2-Lite-Chat-GGUF/resolve/main/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf \
  -O DeepSeek-V2-Lite-Chat.Q4_K_M.gguf
```

启动示例：

```bash
cd "$WORK/src/HybriMoE"

python -m ktransformers.local_chat \
  --model_path deepseek-ai/DeepSeek-V2-Lite-Chat \
  --gguf_path "$WORK/models/DeepSeek-V2-Lite-Chat-GGUF" \
  --cache_size 16 \
  --prefetch_size 0 \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-gpu.yaml
```

如果使用其他模型或 GGUF 量化文件，先确认 `--model_path`、`--gguf_path` 和 optimize rule 三者匹配。

## 9. 常见错误和对应处理

`fatal error: llama.cpp/ggml-common.h: No such file or directory`

原因通常是仍在使用 `llama.cpp=2189fd3...` 的新版布局。按第 3 节回滚到 `e112b610...`，不要继续给新版头文件做一层层转发。

`GGML_TASK_TYPE_COMPUTE` 或 `ggml_internal_get_type_traits` 未定义

说明 `llama.cpp` 仍然太新。必须使用仍保留旧 ggml API 的版本，本次验证版本是 `e112b610a1a75cb7fa8351e1a933e2e7a755a5ce`。

`#error ... pass -fno-finite-math-only`

说明 CMake 里仍有 `-ffast-math`。按第 4 节替换为 `-fno-finite-math-only`。

`git submodule update` 后又编译失败

父仓库 gitlink 会把 `third_party/llama.cpp` 恢复到 `2189fd3...`。回滚后不要再执行 submodule update；如果误执行，重新做第 3 节。

`https://github.com` 连接超时

改用 SSH remote，必要时用 GitHub SSH over 443。见第 2 节。

`NameError: name 'flash_attn_func' is not defined`

这说明模型已经进入 forward，但 `ktransformers/operators/attention.py` 中的 `from flash_attn import flash_attn_func` 导入失败。源码会吞掉这个异常，然后在 Linux Triton/FlashInfer 的 prefill 分支里继续调用 `flash_attn_func`，所以抛出 NameError。

如果不想编译 flash-attn，将两处 `flash_attn_func(...)` 替换为 PyTorch `F.scaled_dot_product_attention(...)`。这个改法是项目 FAQ 给出的 fallback 方向，不需要重新编译 native extension。

## 10. 最小状态检查清单

编译前至少确认：

```bash
cd "$WORK/src/HybriMoE"

git rev-parse HEAD
git -C third_party/llama.cpp rev-parse HEAD
git -C third_party/pybind11 rev-parse HEAD
grep -n "CMAKE_CXX_FLAGS" ktransformers/ktransformers_ext/CMakeLists.txt
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
PY
```

期望状态：

```text
HybriMoE: 508a85b9a43dff37124370d8816e411e74827803
llama.cpp: e112b610a1a75cb7fa8351e1a933e2e7a755a5ce
pybind11: c125cc789ca1be95bcd10479c568d5d260c743ad
CMAKE_CXX_FLAGS includes -fno-finite-math-only
torch 2.5.1+cu124, CUDA 12.4
```
