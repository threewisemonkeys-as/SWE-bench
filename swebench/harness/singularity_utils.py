from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path

HEREDOC_DELIMITER = "EOF_1399519320"  # Different delimiter to avoid conflicts

def run_singularity_command(args, **kwargs):
    """Run a singularity command and return the result"""
    cmd = ["singularity"] + args
    result = subprocess.run(cmd, **kwargs)
    return result

def build_singularity_image(def_file: Path, image_path: Path, force: bool = False):
    """Build a singularity image from a definition file"""
    args = ["build"]
    if force:
        args.append("--force")
    args.extend([str(image_path), str(def_file)])
    
    return run_singularity_command(args, capture_output=True, text=True, check=True)

def start_instance(image_path: Path, instance_name: str, **kwargs):
    """Start a singularity instance"""
    args = ["instance", "start"]
    
    # Add any additional arguments
    for k, v in kwargs.items():
        if k == "bind":
            args.extend(["--bind", v])
        elif k == "home":
            args.extend(["--home", v])
        elif k == "env":
            for env_name, env_value in v.items():
                args.extend(["--env", f"{env_name}={env_value}"])
    
    args.extend([str(image_path), instance_name])
    
    return run_singularity_command(args, capture_output=True, text=True, check=True)

def stop_instance(instance_name: str):
    """Stop a singularity instance"""
    args = ["instance", "stop", instance_name]
    return run_singularity_command(args, capture_output=True, text=True)

def exec_instance(instance_name: str, command: str, **kwargs):
    """Execute a command in a singularity instance"""
    args = ["exec", "instance://" + instance_name]
    
    # Add any additional arguments
    for k, v in kwargs.items():
        if k == "bind":
            args.extend(["--bind", v])
        elif k == "home":
            args.extend(["--home", v])
        elif k == "env":
            for env_name, env_value in v.items():
                args.extend(["--env", f"{env_name}={env_value}"])
    
    args.extend(["/bin/bash", "-c", command])
    
    return run_singularity_command(args, capture_output=True, text=True)

def exec_instance_with_timeout(instance_name: str, command: str, timeout: int|None = 60, **kwargs):
    """Execute a command in a singularity instance with a timeout"""
    result = b''
    exception = None
    timed_out = False
    
    def run_command():
        nonlocal result, exception
        try:
            cmd_result = exec_instance(instance_name, command, **kwargs)
            result = cmd_result.stdout.encode() + cmd_result.stderr.encode()
        except Exception as e:
            exception = e
    
    thread = threading.Thread(target=run_command)
    start_time = time.time()
    thread.start()
    thread.join(timeout)
    
    if exception:
        raise exception
    
    if thread.is_alive():
        # Try to kill the command
        exec_instance(instance_name, f"pkill -f '{command}'")
        timed_out = True
    
    end_time = time.time()
    return result.decode(), timed_out, end_time - start_time

def copy_to_instance(src: Path, dst: Path, instance_name: str):
    """Copy a file to a singularity instance"""
    # Make sure the destination directory exists
    exec_instance(instance_name, f"mkdir -p {dst.parent}")
    
    # Copy the file
    args = ["copy", str(src), f"instance://{instance_name}:{dst}"]
    
    return run_singularity_command(args, capture_output=True, text=True, check=True)

def write_to_instance(data: str, dst: Path, instance_name: str):
    """Write a string to a file in a singularity instance"""
    # Make sure the destination directory exists
    exec_instance(instance_name, f"mkdir -p {dst.parent}")
    
    # Write to file using heredoc
    command = f"cat <<'{HEREDOC_DELIMITER}' > {dst}\n{data}\n{HEREDOC_DELIMITER}"
    
    return exec_instance(instance_name, command)

def list_instances():
    """List all singularity instances"""
    args = ["instance", "list"]
    result = run_singularity_command(args, capture_output=True, text=True)
    
    # Parse the output to get instance names
    lines = result.stdout.strip().split('\n')
    if len(lines) <= 1:  # Just the header line or empty
        return []
    
    instances = []
    for line in lines[1:]:  # Skip header line
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 1:
            instances.append(parts[0])
    
    return instances

def remove_image(image_path: Path):
    """Remove a singularity image"""
    if image_path.exists():
        image_path.unlink()