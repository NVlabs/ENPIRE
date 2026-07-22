# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess

import boto3
from gear.common import get_s3_credentials_from_file
import tyro


def get_s3_bucket(bucket_name: str):
    """Return an S3 bucket resource using local credentials file."""
    creds = get_s3_credentials_from_file()
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=creds["aws_access_key_id"],
        aws_secret_access_key=creds["aws_secret_access_key"],
        region_name=creds["region"],
        endpoint_url=creds["endpoint_url"],
    )
    return s3.Bucket(bucket_name)


def move_s3_prefix(bucket, old_prefix: str, new_prefix: str):
    """Move all objects from old_prefix to new_prefix within the same bucket."""
    for obj in bucket.objects.filter(Prefix=old_prefix):
        old_key = obj.key
        new_key = old_key.replace(old_prefix, new_prefix, 1)
        bucket.copy({"Bucket": bucket.name, "Key": old_key}, new_key)
        bucket.Object(old_key).delete()


def sync_local_session_to_cloud(session_dir: Path, bucket_name: str):
    session_id = session_dir.name
    local_episodes = {p.name for p in session_dir.iterdir() if p.is_dir()}
    print(f"Syncing session to cloud: {session_id} ({len(local_episodes)} episodes)")

    # Upload session
    cmd = ["gear", "data", "upload", str(session_dir), f"{bucket_name}/{session_id}"]
    subprocess.run(cmd, check=True)

    # Compare local and cloud episodes
    bucket = get_s3_bucket(bucket_name)
    cloud_episodes = {
        obj.key.removeprefix(f"{session_id}/").split("/", 1)[0]
        for obj in bucket.objects.filter(Prefix=f"{session_id}/")
    }

    # Discard cloud episodes that are not present locally
    missing_in_local = cloud_episodes - local_episodes
    if len(missing_in_local) > 0:
        print("Found episodes in cloud not present locally, moving to trash:")
        for episode_id in missing_in_local:
            move_s3_prefix(
                bucket, f"{session_id}/{episode_id}/", f"trash/{session_id}/{episode_id}/"
            )
            print(f"- {episode_id}")


def main(session_ids: list[str], /, data_dir: str = "data", bucket_name: str = "GearYAMRawDataV1"):
    for session_id in session_ids:
        session_dir = Path(data_dir) / session_id
        assert session_dir.exists(), f"Session directory {session_dir} does not exist"
        sync_local_session_to_cloud(session_dir, bucket_name)
    print("All sessions synced successfully.")


if __name__ == "__main__":
    tyro.cli(main)
