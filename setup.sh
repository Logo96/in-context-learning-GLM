
ENV_YML="env1.yml"
PIP_TXT="env2.txt"
CONDA_DIR="$HOME/miniconda3"

# Extract environment name from env1.yml using grep/sed
ENV_NAME=$(grep "^name:" "$ENV_YML" | sed 's/name:[[:space:]]*//')

# Step 1: Install Miniconda if not already installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$CONDA_DIR"
    export PATH="$CONDA_DIR/bin:$PATH"
    echo "Miniconda installed."
else
    echo "Conda already installed."
fi

# Ensure conda is available
export PATH="$CONDA_DIR/bin:$PATH"
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Step 1.5: Install tmux if not already installed
# Step 1.5: Install tmux if not already installed (Docker-safe)
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux (Docker-safe)..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y tmux || echo "tmux install failed, but continuing..."
else
    echo "tmux already installed."
fi


# Step 2: Create conda environment if it doesn't exist
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating conda environment from $ENV_YML..."
    conda env create -f "$ENV_YML"
fi

# Step 3: Activate the environment
echo "Activating environment '$ENV_NAME'..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Step 4: Install pip dependencies
if [ -f "$PIP_TXT" ]; then
    echo "Installing pip packages from $PIP_TXT..."
    pip install -r "$PIP_TXT"
else
    echo "env2.txt not found. Skipping pip install."
fi
