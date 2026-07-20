#!/usr/bin/env bash
#
# One-shot environment setup for training Turt on a Colab/Kaggle GPU VM
# (see docs/CLOUD.md). Installs Vulkan + the NVIDIA ICD, Deno, Node 22 + pnpm,
# project deps, points checkpoints/ at Google Drive when it is mounted (so
# they outlive the VM), and validates the GPU kernels against the CPU engine.
#
# Usage (from a notebook cell, after cloning the repo):
#   !bash /content/turt/scripts/colab-setup.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== installing Vulkan ==="
apt-get -qq update
apt-get -qq install -y libvulkan1 vulkan-tools > /dev/null
# Colab's minimal NVIDIA driver install often lacks the Vulkan ICD; add it.
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
apt-get -qq install -y "libnvidia-gl-${DRIVER}" > /dev/null || true
mkdir -p /usr/share/vulkan/icd.d
cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}
EOF
if ! vulkaninfo --summary 2>/dev/null | grep -qi nvidia; then
  echo "ERROR: Vulkan does not see the NVIDIA GPU — WebGPU would fall back to CPU."
  vulkaninfo --summary || true
  exit 1
fi

echo "=== installing Deno + Node 22 + pnpm ==="
command -v deno > /dev/null || curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh > /dev/null
node --version 2>/dev/null | grep -qE "^v(2[2-9]|[3-9][0-9])" || {
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - > /dev/null
  apt-get -qq install -y nodejs > /dev/null
}
command -v pnpm > /dev/null || npm install -g pnpm --silent
pnpm install --silent

# Checkpoints must outlive the VM: keep them on Drive when it is mounted.
if [ -d /content/drive/MyDrive ]; then
  mkdir -p /content/drive/MyDrive/turt-checkpoints
  rm -rf checkpoints
  ln -sfn /content/drive/MyDrive/turt-checkpoints checkpoints
  echo "checkpoints/ -> Drive (survives session resets)"
else
  echo "WARNING: Drive not mounted — checkpoints will die with this VM."
fi

echo "=== validating GPU kernels (bit-for-bit vs CPU engine) ==="
pnpm gpu-gpt
echo "=== setup OK ==="
