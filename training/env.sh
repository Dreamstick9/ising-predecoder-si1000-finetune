# Paths used by the scripts in this folder. Source it, do not execute it.
#   QEC_ROOT            working directory holding venv/, models/, runs/ and logs/
#   ISING_DECODING_DIR  patched checkout of https://github.com/NVIDIA/Ising-Decoding
export P=${QEC_ROOT:-$HOME/qec-ising}
export REPO=${ISING_DECODING_DIR:-$P/src/Ising-Decoding-main}
export PATH=$P/venv/bin:$PATH
