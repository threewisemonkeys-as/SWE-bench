# Singularity Definition file templates
_DEFINITION_BASE = r"""
Bootstrap: docker
From: ubuntu:22.04

%post
    export DEBIAN_FRONTEND=noninteractive
    export TZ=Etc/UTC
    
    apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libffi-dev \
    libtiff-dev \
    python3 \
    python3-pip \
    python-is-python3 \
    jq \
    curl \
    locales \
    locales-all \
    tzdata
    
    # Download and install conda
    wget 'https://repo.anaconda.com/miniconda/Miniconda3-{conda_version}-Linux-{conda_arch}.sh' -O miniconda.sh
    bash miniconda.sh -b -p /opt/miniconda3
    
    # Add conda to PATH
    echo 'export PATH=/opt/miniconda3/bin:$PATH' >> $SINGULARITY_ENVIRONMENT
    
    # Add conda to shell startup scripts
    /opt/miniconda3/bin/conda init --all
    /opt/miniconda3/bin/conda config --append channels conda-forge
    
    # Add nonroot user
    adduser --disabled-password --gecos 'dog' nonroot

%environment
    export PATH=/opt/miniconda3/bin:$PATH

%startscript
    exec bash
"""

_DEFINITION_ENV = r"""
Bootstrap: localimage
From: {base_image}

%files
    ./setup_env.sh /root/setup_env.sh

%post
    sed -i -e 's/\r$//' /root/setup_env.sh
    chmod +x /root/setup_env.sh
    source ~/.bashrc && /root/setup_env.sh
    
    # Automatically activate the testbed environment
    echo "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed" > /root/.bashrc

%environment
    export SINGULARITY_WORKDIR=/testbed/

%startscript
    exec bash
"""

_DEFINITION_INSTANCE = r"""
Bootstrap: localimage
From: {env_image}

%files
    ./setup_repo.sh /root/setup_repo.sh

%post
    sed -i -e 's/\r$//' /root/setup_repo.sh
    bash /root/setup_repo.sh

%environment
    export SINGULARITY_WORKDIR=/testbed/

%startscript
    exec bash
"""

def get_definition_base(platform, arch):
    """Generate base definition file"""
    if arch == "arm64":
        conda_arch = "aarch64"
    else:
        conda_arch = arch
    conda_version = "py311_23.11.0-2"  # Default version
    
    return _DEFINITION_BASE.format(conda_arch=conda_arch, conda_version=conda_version)

def get_definition_env(platform, arch):
    """Generate environment definition file"""
    return _DEFINITION_ENV

def get_definition_instance(platform):
    """Generate instance definition file"""
    return _DEFINITION_INSTANCE