# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import subprocess
import zipfile


def download_asset_github(repo, tag, pattern, install_dir=None):
    # Construct the gh command
    if tag:
        gh_command = f"gh release download {tag} --repo {repo} -p '{pattern}'"
    else:
        gh_command = f"gh release download --repo {repo} -p '{pattern}'"

    print(f"Executing command: {gh_command}")

    # Execute the gh command
    try:
        subprocess.run(gh_command, shell=True, check=True)
        print(f"Asset(s) matching '{pattern}' downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading asset: {e}")
        sys.exit(1)

    # Add output directory to the command
    if install_dir and pattern.endswith(".whl"):
        # Extract the contents of the wheel file
        with zipfile.ZipFile(pattern, 'r') as zip_ref:
            zip_ref.extractall(install_dir)


def download_asset_wget(url, pattern):
    try:
        result = subprocess.run(['wget', url + pattern],
                                capture_output=True,
                                text=True,
                                check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print(f"wget output: {e.output}")


def main():
    # Read the entry from a JSON file
    with open('.cudaq_version', 'r') as json_file:
        assets = json.load(json_file)

    for name, info in assets.items():
        if "tag" in info:
            download_asset_github(info["repository"], info["tag"],
                                  info["pattern"], info.get("install_dir"))
        if "url" in info:
            download_asset_wget(info["url"], info["pattern"])


if __name__ == "__main__":
    main()
