"""Upload a gif to ImageOptim API to convert them to WebM."""

# Instructions in https://imageoptim.com/api/ungif

import argparse
from pathlib import Path
import glob

# might be interesting to use aiohttp instead (just to learn)
# https://docs.aiohttp.org/en/stable/client_quickstart.html#post-a-multipart-encoded-file
import requests


def set_config():

    parser = argparse.ArgumentParser(description="ImageOptim configuration.")
    parser.add_argument("--format", default="webm")
    parser.add_argument("--username", required=True)
    parser.add_argument("--gifs-dir", required=True)

    config = parser.parse_args()
    url = f"https://im2.io/{config.username}/format={config.format}"
    return url, config.gifs_dir


def convert_all_gifs():

    url, gifs_dir = set_config()

    gif_paths = [path for path in gifs_dir.rglob("*.gif")]

    for gif_path in gif_paths:

        webm_path = gif_path.with_suffix(".webm")

        if webm_path.is_file():
            continue

        # NOTE: Maximum gif size is 50MB

        with open(gif_path, "rb") as gif:

            response = requests.post(url, files={"file": gif}, timeout=120)

            promised = int(response.headers["Content-Length"])
            downloaded = len(response.content)

            if promised != downloaded:

                print(f"Truncated processing of {gif_path}")
                print(
                    f"Downloaded {downloaded} of {promised} bytes... Please try again!"
                )
                continue

            with open(webm_path, "wb") as file:
                file.write(response.content)

            print(f"Got new animation! --> {webm_path}")


if __name__ == "__main__":

    convert_all_gifs()
