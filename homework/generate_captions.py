from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import (
    draw_detections,
    extract_frame_info,
    extract_kart_objects,
    extract_track_info,
)


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate captions for a specific view.

    Returns a list of caption strings, one per fact about the scene.
    Caption formats are matched exactly to the candidates in all_mc_qas.json:
        "{kart_name} is the ego car."
        "There are {n} karts in the scene."
        "The track is {track_name}."
        "{kart_name} is in front of the ego car."
        "{kart_name} is behind the ego car."
        "{kart_name} is left of the ego car."
        "{kart_name} is right of the ego car."
    """
    captions = []

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    ego_kart = next((k for k in karts if k["is_center_kart"]), None)
    other_karts = [k for k in karts if not k["is_center_kart"]]

    img_cx = img_width / 2.0
    img_cy = img_height / 2.0

    def classify_lr(cx):
        return "left" if cx < img_cx else "right"

    def classify_fb(cy):
        return "in front of" if cy < img_cy else "behind"

    # 1. Ego car caption
    if ego_kart is not None:
        captions.append(f"{ego_kart['kart_name']} is the ego car.")

    # 2. Kart count caption
    captions.append(f"There are {len(karts)} karts in the scene.")

    # 3. Track caption
    captions.append(f"The track is {track_name}.")

    # 4. Relative position captions for each visible non-ego kart
    for kart in other_karts:
        name = kart["kart_name"]
        cx, cy = kart["center"]
        lr = classify_lr(cx)
        fb = classify_fb(cy)

        # front/behind caption
        captions.append(f"{name} is {fb} the ego car.")

        # left/right caption
        captions.append(f"{name} is {lr} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_all(
    data_dir: str = "data/train",
    output_file: str = "data/train/train_captions.json",
):
    """
    Walk every *_info.json in data_dir, generate captions for every view,
    and write a single combined JSON file consumable by CaptionDataset.

    Usage:
        python -m homework.generate_captions generate_all \\
            --data_dir data/train \\
            --output_file data/train/train_captions.json
    """
    import json

    data_path = Path(data_dir)
    info_files = sorted(data_path.glob("*_info.json"))

    if not info_files:
        print(f"No *_info.json files found in {data_dir}")
        return

    all_captions = []

    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        image_files = sorted(info_file.parent.glob(f"{base_name}_*_im.jpg"))

        for image_file in image_files:
            _, view_index = extract_frame_info(str(image_file))

            try:
                captions = generate_caption(str(info_file), view_index)
            except Exception as e:
                print(f"Skipping {info_file} view {view_index}: {e}")
                continue

            # Store path relative to the data root (one level up from data_dir)
            rel_image = image_file.relative_to(data_path.parent)

            for caption in captions:
                all_captions.append({
                    "image_file": str(rel_image),
                    "caption": caption,
                })

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_captions, f, indent=2)

    print(f"Wrote {len(all_captions)} captions to {output_path}")


"""
Usage:

Visualize captions for one frame:
    python -m homework.generate_captions check \\
        --info_file data/train/00000_info.json \\
        --view_index 0

Generate the full training caption dataset:
    python -m homework.generate_captions generate_all \\
        --data_dir data/train \\
        --output_file data/train/train_captions.json
"""


def main():
    fire.Fire({"check": check_caption, "generate_all": generate_all})


if __name__ == "__main__":
    main()