git config --global safe.directory '*'
git config --global core.editor "code --wait"
git config --global pager.branch false

# Set AZCOPY concurrency to auto
echo "export AZCOPY_CONCURRENCY_VALUE=AUTO" >> ~/.zshrc
echo "export AZCOPY_CONCURRENCY_VALUE=AUTO" >> ~/.bashrc

# Activate conda by default
echo ". /home/vscode/miniconda3/bin/activate" >> ~/.zshrc
echo ". /home/vscode/miniconda3/bin/activate" >> ~/.bashrc

# Use llava environment by default
echo "conda activate llava" >> ~/.zshrc
echo "conda activate llava" >> ~/.bashrc

# Keep model caches outside the repo.
echo 'export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"' >> ~/.bashrc
echo 'export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"' >> ~/.zshrc
echo 'export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"' >> ~/.zshrc

# Add dotnet to PATH
echo 'export PATH="$PATH:$HOME/.dotnet"' >> ~/.bashrc
echo 'export PATH="$PATH:$HOME/.dotnet"' >> ~/.zshrc

# Create the llava environment with the pinned setup used in this workspace.
export LLAVA_INSTALL_CUDA_COMPILER=1
bash ./scripts/setup_llava_env.sh

echo "postCreateCommand.sh COMPLETE!"
