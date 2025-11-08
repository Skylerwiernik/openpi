#!/usr/bin/env python3
"""
Convert LeRobot weights to OpenPI format by stripping the 'model.' prefix from all keys.

This script loads a safetensors file from LeRobot (which has 'model.' prefix on all keys)
and saves a new safetensors file with the prefix removed for compatibility with OpenPI.

It also transforms the config files to use OpenPI's observation format with the following mappings:
  - observation.images.scene -> observation/image
  - observation.images.gripper -> observation/wrist_image

Usage:
    python scripts/convert_lerobot_weights.py --input_path /path/to/lerobot/model.safetensors --output_dir /path/to/output
"""

import json
import os
import shutil
from pathlib import Path

import safetensors
import safetensors.torch
import torch
import tyro


def _transform_config_keys(config: dict) -> dict:
    """
    Transform LeRobot observation keys to OpenPI format.

    Maps:
      - observation.images.scene -> observation/image
      - observation.images.gripper -> observation/wrist_image
      - observation.state -> observation/state
    """
    # Transform input_features keys
    if "input_features" in config:
        new_input_features = {}
        for key, value in config["input_features"].items():
            if key == "observation.images.scene":
                new_input_features["observation/image"] = value
            elif key == "observation.images.gripper":
                new_input_features["observation/wrist_image"] = value
            elif key == "observation.state":
                new_input_features["observation/state"] = value
            else:
                new_input_features[key] = value
        config["input_features"] = new_input_features

    # Transform output_features keys (usually just 'action')
    if "output_features" in config:
        new_output_features = {}
        for key, value in config["output_features"].items():
            if key == "action":
                new_output_features["actions"] = value
            else:
                new_output_features[key] = value
        config["output_features"] = new_output_features

    return config


def _transform_preprocessor_json(preprocessor: dict) -> dict:
    """
    Transform preprocessor JSON to use OpenPI observation keys.

    Updates the features dict in the normalizer_processor step.
    """
    if "steps" in preprocessor:
        for step in preprocessor["steps"]:
            if step.get("registry_name") == "normalizer_processor":
                if "config" in step and "features" in step["config"]:
                    new_features = {}
                    for key, value in step["config"]["features"].items():
                        if key == "observation.images.scene":
                            new_features["observation/image"] = value
                        elif key == "observation.images.gripper":
                            new_features["observation/wrist_image"] = value
                        elif key == "observation.state":
                            new_features["observation/state"] = value
                        else:
                            new_features[key] = value
                    step["config"]["features"] = new_features

    return preprocessor


def convert_lerobot_weights(input_path: str, output_dir: str):
    """
    Convert LeRobot weights to OpenPI format.

    Args:
        input_path: Path to the input safetensors file (or directory containing model.safetensors)
        output_dir: Directory to save the converted weights
    """
    # Handle input path - can be file or directory
    input_path = Path(input_path)
    if input_path.is_dir():
        input_file = input_path / "model.safetensors"
        input_dir = input_path
    else:
        input_file = input_path
        input_dir = input_path.parent

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading weights from: {input_file}")

    # Load the safetensors file
    state_dict = {}
    with safetensors.safe_open(str(input_file), framework="pt") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    print(f"Loaded {len(state_dict)} tensors")

    # Strip 'model.' prefix from all keys
    converted_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key[6:]  # Remove 'model.' prefix (6 characters)
            converted_state_dict[new_key] = value
            print(f"  {key} -> {new_key}")
        else:
            converted_state_dict[key] = value
            print(f"  {key} (no change)")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save converted weights
    output_file = output_dir / "model.safetensors"
    print(f"\nSaving converted weights to: {output_file}")
    safetensors.torch.save_file(converted_state_dict, str(output_file))

    # Copy and transform config files
    print("\nTransforming configuration files...")

    # Handle config.json
    config_file = input_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        config = _transform_config_keys(config)
        output_config_file = output_dir / "config.json"
        with open(output_config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Transformed config.json")

    # Handle policy_preprocessor.json
    preprocessor_file = input_dir / "policy_preprocessor.json"
    if preprocessor_file.exists():
        with open(preprocessor_file) as f:
            preprocessor = json.load(f)
        preprocessor = _transform_preprocessor_json(preprocessor)
        output_preprocessor_file = output_dir / "policy_preprocessor.json"
        with open(output_preprocessor_file, "w") as f:
            json.dump(preprocessor, f, indent=2)
        print(f"Transformed policy_preprocessor.json")

    # Copy other files without transformation
    files_to_copy = [
        "train_config.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]

    for filename in files_to_copy:
        src = input_dir / filename
        if src.exists():
            dst = output_dir / filename
            print(f"Copying {filename}")
            shutil.copy2(src, dst)

    # Copy assets directory if it exists
    assets_dir = input_dir / "assets"
    if assets_dir.exists():
        output_assets_dir = output_dir / "assets"
        print(f"Copying assets directory...")
        shutil.copytree(assets_dir, output_assets_dir, dirs_exist_ok=True)

    print(f"\nConversion complete! Output saved to: {output_dir}")
    print(f"Total tensors converted: {len(converted_state_dict)}")


if __name__ == "__main__":
    tyro.cli(convert_lerobot_weights)
