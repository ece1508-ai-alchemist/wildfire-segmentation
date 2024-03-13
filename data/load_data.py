import os
import pathlib

from rich.console import Console

console = Console()

DATA_FOLDER = pathlib.Path(__file__).parent

PARENT_DIR = pathlib.Path(__file__).parent

CACHE_DIR = PARENT_DIR / ".cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR.absolute())


PRE_POST_FIRE_DIR = PARENT_DIR / "pre_post_fire"
PRE_POST_FIRE_DIR.mkdir(parents=True, exist_ok=True)

POST_FIRE_DIR = PARENT_DIR / "post_fire"
POST_FIRE_DIR.mkdir(parents=True, exist_ok=True)


def is_empty_dir(path: pathlib.Path) -> bool:
    return not any(path.iterdir())


def load() -> None:
    from datasets import load_dataset

    with console.status("Loading data..."):
        if is_empty_dir(PRE_POST_FIRE_DIR):
            console.log("Loading pre post fire dataset")
            pre_post_fire_dataset = load_dataset(
                "DarthReca/california_burned_areas", name="pre-post-fire", trust_remote_code=True
            )
            console.log("Saving pre post fire dataset")
            pre_post_fire_dataset.save_to_disk(PRE_POST_FIRE_DIR)
            console.log(f"pre post fire dataset was saved to {PRE_POST_FIRE_DIR.absolute()}")

        if is_empty_dir(POST_FIRE_DIR):
            console.log("Loading post fire dataset")
            post_fire_dataset = load_dataset(
                "DarthReca/california_burned_areas", name="post-fire", trust_remote_code=True
            )
            console.log("Saving post fire dataset")
            post_fire_dataset.save_to_disk(POST_FIRE_DIR)
            console.log(f"post fire dataset was saved to {POST_FIRE_DIR.absolute()}")


if __name__ == "__main__":
    load()
    print("Datasets were loaded and cached successfully, happy training \U0001f60a")
