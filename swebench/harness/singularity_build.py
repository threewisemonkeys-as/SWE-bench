from __future__ import annotations

import logging
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from swebench.harness.constants import (
    DOCKER_USER,
    BASE_IMAGE_BUILD_DIR,
    ENV_IMAGE_BUILD_DIR,
    INSTANCE_IMAGE_BUILD_DIR,
    MAP_REPO_VERSION_TO_SPECS,
    UTF8,
)
from swebench.harness.test_spec import (
    get_test_specs_from_dataset,
    make_test_spec,
    TestSpec
)
from swebench.harness.singularity_utils import (
    build_singularity_image,
    remove_image,
    start_instance,
    stop_instance,
)
from swebench.harness.singularity_definition_files import (
    get_definition_base,
    get_definition_env,
    get_definition_instance,
)

ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

class BuildImageError(Exception):
    def __init__(self, image_name, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.image_name = image_name
        self.log_path = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Error building image {self.image_name}: {self.super_str}\n"
            f"Check ({self.log_path}) for more information."
        )

def setup_logger(instance_id: str, log_file: Path, mode="w"):
    """Set up a logger for build process"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{instance_id}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode, encoding=UTF8)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    return logger

def close_logger(logger):
    """Close a logger"""
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

def build_image(
        image_name: str,
        setup_scripts: dict,
        definition: str,
        build_dir: Path,
        force_rebuild: bool = False
    ):
    """Build a singularity image with the given definition and setup scripts"""
    logger = setup_logger(image_name, build_dir / "build_image.log")
    logger.info(
        f"Building image {image_name}\n"
        f"Using definition:\n{definition}\n"
        f"Adding ({len(setup_scripts)}) setup scripts to image build repo"
    )

    for setup_script_name, setup_script in setup_scripts.items():
        logger.info(f"[SETUP SCRIPT] {setup_script_name}:\n{setup_script}")
    
    try:
        # Write setup scripts
        for setup_script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)

        # Write definition file
        definition_path = build_dir / "Singularity.def"
        with open(definition_path, "w") as f:
            f.write(definition)

        # Build the image
        image_path = build_dir / f"{image_name.replace(':', '_')}.sif"
        logger.info(f"Building singularity image {image_name} in {build_dir}")
        
        if image_path.exists() and not force_rebuild:
            logger.info(f"Image {image_name} already exists, skipping build.")
            return image_path
            
        result = build_singularity_image(definition_path, image_path, force=force_rebuild)
        logger.info(result.stdout)
        logger.info("Image built successfully!")
        return image_path
    
    except Exception as e:
        logger.error(f"Error building image {image_name}: {e}")
        logger.error(traceback.format_exc())
        raise BuildImageError(image_name, str(e), logger) from e
    
    finally:
        close_logger(logger)

def build_container(
        test_spec: TestSpec,
        run_id: str,
        logger: logging.Logger,
        force_rebuild: bool = False
    ):
    """
    Builds the instance image for the given test spec and creates a Singularity instance.
    
    Args:
        test_spec (TestSpec): Test spec to build the instance image and container for
        run_id (str): Run ID identifying process, used for the instance name
        logger (logging.Logger): Logger to use for logging the build process
        force_rebuild (bool): Whether to force rebuild the image even if it already exists
    
    Returns:
        str: Name of the created instance
    """
    # Build base and env images
    base_image_path = build_base_image(test_spec, logger, force_rebuild)
    env_image_path = build_env_image(test_spec, base_image_path, logger, force_rebuild)
    
    # Build instance image
    instance_build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
    instance_build_dir.mkdir(parents=True, exist_ok=True)
    
    # Write setup_repo.sh script
    setup_repo_path = instance_build_dir / "setup_repo.sh"
    with open(setup_repo_path, "w") as f:
        f.write(test_spec.install_repo_script)
    
    # Make definition file from template
    definition = test_spec.instance_definition.format(env_image=env_image_path)
    definition_path = instance_build_dir / "Singularity.def"
    with open(definition_path, "w") as f:
        f.write(definition)
    
    # Build instance image
    instance_image_path = instance_build_dir / f"{test_spec.instance_image_key.replace(':', '_')}.sif"
    if not instance_image_path.exists() or force_rebuild:
        logger.info(f"Building instance image {test_spec.instance_image_key}")
        try:
            build_singularity_image(definition_path, instance_image_path, force=force_rebuild)
        except Exception as e:
            logger.error(f"Error building instance image: {e}")
            raise BuildImageError(test_spec.instance_id, str(e), logger)
    else:
        logger.info(f"Instance image {test_spec.instance_image_key} already exists")
    
    # Start a new instance
    instance_name = f"{test_spec.get_instance_container_name(run_id)}"
    
    try:
        # Get configurations for how container should be created
        config = MAP_REPO_VERSION_TO_SPECS[test_spec.repo][test_spec.version]
        user = "nonroot" if config.get("execute_test_as_nonroot", False) else "root"
        
        # Create workdir if it doesn't exist
        workdir = Path(f"/tmp/swebench/{instance_name}")
        workdir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting Singularity instance {instance_name}")
        start_instance(
            instance_image_path, 
            instance_name,
            bind=f"{workdir}:/testbed",
            env={"USER": user}
        )
        
        logger.info(f"Instance {instance_name} started")
        return instance_name
    
    except Exception as e:
        logger.error(f"Error creating instance for {test_spec.instance_id}: {e}")
        logger.info(traceback.format_exc())
        stop_instance(instance_name)
        raise BuildImageError(test_spec.instance_id, str(e), logger)
    
def build_base_image(test_spec: TestSpec, logger: logging.Logger, force_rebuild: bool = False):
    """Build a base Singularity image for the given test spec and return its path"""
    # Create build directory
    build_dir = BASE_IMAGE_BUILD_DIR / test_spec.base_image_key.replace(":", "__")
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Define image path
    image_path = build_dir / f"{test_spec.base_image_key.replace(':', '_')}.sif"
    
    # Check if image already exists
    if image_path.exists() and not force_rebuild:
        logger.info(f"Base image {test_spec.base_image_key} already exists at {image_path}")
        return image_path
    
    logger.info(f"Building base image {test_spec.base_image_key}")
    
    # Generate definition file from template based on platform and architecture
    definition_content = get_definition_base(test_spec.platform, test_spec.arch)
    
    # Write definition file
    definition_path = build_dir / "Singularity.def"
    with open(definition_path, "w") as f:
        f.write(definition_content)
    
    # Build the image
    try:
        build_singularity_image(definition_path, image_path, force=force_rebuild)
        logger.info(f"Base image built successfully at {image_path}")
        return image_path
    except Exception as e:
        logger.error(f"Error building base image: {e}")
        raise BuildImageError(test_spec.base_image_key, str(e), logger)


def build_env_image(test_spec: TestSpec, base_image_path: Path, logger: logging.Logger, force_rebuild: bool = False):
    """Build an environment Singularity image for the given test spec and return its path"""
    # Create build directory
    build_dir = ENV_IMAGE_BUILD_DIR / test_spec.env_image_key.replace(":", "__")
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Define image path
    image_path = build_dir / f"{test_spec.env_image_key.replace(':', '_')}.sif"
    
    # Check if image already exists
    if image_path.exists() and not force_rebuild:
        logger.info(f"Environment image {test_spec.env_image_key} already exists at {image_path}")
        return image_path
    
    logger.info(f"Building environment image {test_spec.env_image_key}")
    
    # Write setup_env.sh script
    setup_env_path = build_dir / "setup_env.sh"
    with open(setup_env_path, "w") as f:
        f.write(test_spec.setup_env_script)
    
    # Make definition file from template
    definition = test_spec.env_definition.format(
        base_image=base_image_path,
        platform=test_spec.platform,
        arch=test_spec.arch
    )
    
    definition_path = build_dir / "Singularity.def"
    with open(definition_path, "w") as f:
        f.write(definition)
    
    # Build the image
    try:
        build_singularity_image(definition_path, image_path, force=force_rebuild)
        logger.info(f"Environment image built successfully at {image_path}")
        return image_path
    except Exception as e:
        logger.error(f"Error building environment image: {e}")
        raise BuildImageError(test_spec.env_image_key, str(e), logger)


def build_env_images(
        dataset: list,
        force_rebuild: bool = False,
        max_workers: int = 4
    ):
    """Build all environment images required for the dataset"""
    test_specs = get_test_specs_from_dataset(dataset)
    
    # First, build all base images
    base_images = {}
    for spec in test_specs:
        if spec.base_image_key not in base_images:
            logger = setup_logger(spec.base_image_key, 
                                 BASE_IMAGE_BUILD_DIR / spec.base_image_key.replace(":", "__") / "build.log")
            try:
                base_image_path = build_base_image(spec, logger, force_rebuild)
                base_images[spec.base_image_key] = base_image_path
            finally:
                close_logger(logger)
    
    print(f"Built {len(base_images)} base images")
    
    # Now build environment images in parallel
    env_images = {}
    successful, failed = [], []
    
    with tqdm(total=len(test_specs), smoothing=0, desc="Building environment images") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for each env image to build
            futures = {}
            for spec in test_specs:
                if spec.env_image_key not in env_images:
                    logger = setup_logger(spec.env_image_key, 
                                         ENV_IMAGE_BUILD_DIR / spec.env_image_key.replace(":", "__") / "build.log")
                    futures[executor.submit(
                        build_env_image, 
                        spec, 
                        base_images[spec.base_image_key], 
                        logger, 
                        force_rebuild
                    )] = (spec, logger)
            
            # Wait for completion
            for future in as_completed(futures):
                pbar.update(1)
                spec, logger = futures[future]
                try:
                    env_image_path = future.result()
                    env_images[spec.env_image_key] = env_image_path
                    successful.append(spec.env_image_key)
                except Exception as e:
                    print(f"Error building environment image {spec.env_image_key}: {e}")
                    failed.append(spec.env_image_key)
                finally:
                    close_logger(logger)
    
    if not failed:
        print("All environment images built successfully")
    else:
        print(f"{len(failed)} environment images failed to build")
    
    return successful, failed