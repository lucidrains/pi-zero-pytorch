#!/bin/bash

# Ensure checkpoints directory exists
mkdir -p checkpoints/pi0_base

# Download the pi0-base weights from Hugging Face
# This requires the huggingface_hub to be installed (pip install huggingface_hub)
hf download lerobot/pi0_base --local-dir checkpoints/pi0_base
