"""Upload a gif to ImageOptim API to convert them to WebM."""

# Instructions in https://imageoptim.com/api/ungif

from pathlib import Path
import glob

import requests

OPTIONS = "format=webm"  # comma-separated options
USERNAME = "dlbggtxxth"

URL = f"https://im2.io/{USERNAME}/{OPTIONS}"

BASE_DIR = Path(__file__).parent / "figures"


def convert_all_gifs():

    gif_paths = [path for path in BASE_DIR.rglob("*.gif")]

    for gif_path in gif_paths:

        webm_path = gif_path.with_suffix(".webm")

        if webm_path.is_file():
            continue

        # NOTE: Maximum gif size is 50MB

        with open(gif_path, "rb") as gif:

            response = requests.post(URL, files={"file": gif}, timeout=120)

            promised = int(response.headers["Content-Length"])
            downloaded = len(response.content)

            if promised != downloaded:

                print(f"Truncated processing of {gif_path}")
                print(f"Downloaded {downloaded} of {promised} bytes... Please try again!")
                continue

            with open(webm_path, "wb") as file:
                file.write(response.content)

            print(f"Got new animation! --> {webm_path}")


if __name__ == "__main__":

    convert_all_gifs()
