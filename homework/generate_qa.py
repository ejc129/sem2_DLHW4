import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    with open(info_path) as f:
        info = json.load(f)

    # ego kart is identified by track_id == 0 in the detections
    # kart names are stored in info["karts"] keyed by instance id
    karts_meta = info.get("karts", [])  # e.g. ["tux", "gnu", ...] indexed by track_id

    if view_index >= len(info["detections"]):
        return []

    frame_detections = info["detections"][view_index]

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    img_cx = img_width / 2.0
    img_cy = img_height / 2.0

    kart_objects = []

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        # Only process karts (class_id == 1)
        if class_id != 1:
            continue

        # Scale bounding box to image dimensions
        x1_s = x1 * scale_x
        y1_s = y1 * scale_y
        x2_s = x2 * scale_x
        y2_s = y2 * scale_y

        # Filter out karts entirely outside the image
        if x2_s < 0 or x1_s > img_width or y2_s < 0 or y1_s > img_height:
            continue

        # Filter out karts whose bounding box is too small
        if (x2_s - x1_s) < min_box_size or (y2_s - y1_s) < min_box_size:
            continue

        center_x = (x1_s + x2_s) / 2.0
        center_y = (y1_s + y2_s) / 2.0

        # Look up kart name by track_id; track_id == 0 is the ego kart
        kart_name = karts_meta[track_id] if track_id < len(karts_meta) else f"kart_{track_id}"

        kart_objects.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y),
            "is_center_kart": False,  # will be set below
        })

    if not kart_objects:
        return []

    # The ego kart has track_id == 0; mark it as the center kart.
    # If no track_id==0 is visible, fall back to the kart whose center is
    # closest to the image center.
    ego_candidates = [k for k in kart_objects if k["instance_id"] == 0]
    if ego_candidates:
        ego_candidates[0]["is_center_kart"] = True
    else:
        closest = min(
            kart_objects,
            key=lambda k: (k["center"][0] - img_cx) ** 2 + (k["center"][1] - img_cy) ** 2,
        )
        closest["is_center_kart"] = True

    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path) as f:
        info = json.load(f)

    # The track name is stored under the "track" key
    return info["track"]


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    qa_pairs = []

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    # Identify ego kart
    ego_kart = next((k for k in karts if k["is_center_kart"]), None)
    # All non-ego karts that are visible
    other_karts = [k for k in karts if not k["is_center_kart"]]

    # Image center — used to determine front/back and left/right relative to ego
    img_cx = img_width / 2.0
    img_cy = img_height / 2.0

    # ------------------------------------------------------------------ #
    # 1. Ego car question
    # ------------------------------------------------------------------ #
    if ego_kart is not None:
        qa_pairs.append({
            "question": "What kart is the ego car?",
            "answer": ego_kart["kart_name"],
        })

    # ------------------------------------------------------------------ #
    # 2. Total karts visible in this view
    # ------------------------------------------------------------------ #
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts)),
    })

    # ------------------------------------------------------------------ #
    # 3. Track name
    # ------------------------------------------------------------------ #
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name,
    })

    # ------------------------------------------------------------------ #
    # Helper: classify a kart's position relative to the ego/image center.
    #
    # Convention (matching balanced_qa_pairs.json):
    #   - x < img_cx  → "left"   (kart appears on the left side of the image)
    #   - x > img_cx  → "right"
    #   - y < img_cy  → "front"  (kart is higher up / further ahead in the scene)
    #   - y > img_cy  → "back"
    # ------------------------------------------------------------------ #
    def classify_lr(cx):
        return "left" if cx < img_cx else "right"

    def classify_fb(cy):
        return "front" if cy < img_cy else "back"

    # ------------------------------------------------------------------ #
    # 4. Relative position questions for each visible non-ego kart
    # ------------------------------------------------------------------ #
    for kart in other_karts:
        name = kart["kart_name"]
        cx, cy = kart["center"]
        lr = classify_lr(cx)
        fb = classify_fb(cy)

        # "Is X to the left or right of the ego car?"
        qa_pairs.append({
            "question": f"Is {name} to the left or right of the ego car?",
            "answer": lr,
        })

        # "Is X in front of or behind the ego car?"
        qa_pairs.append({
            "question": f"Is {name} in front of or behind the ego car?",
            "answer": fb,
        })

        # "Where is X relative to the ego car?"
        qa_pairs.append({
            "question": f"Where is {name} relative to the ego car?",
            "answer": f"{fb} and {lr}",
        })

    # ------------------------------------------------------------------ #
    # 5. Counting questions
    # ------------------------------------------------------------------ #
    left_count  = sum(1 for k in other_karts if classify_lr(k["center"][0]) == "left")
    right_count = sum(1 for k in other_karts if classify_lr(k["center"][0]) == "right")
    front_count = sum(1 for k in other_karts if classify_fb(k["center"][1]) == "front")
    back_count  = sum(1 for k in other_karts if classify_fb(k["center"][1]) == "back")

    qa_pairs.append({"question": "How many karts are to the left of the ego car?",  "answer": str(left_count)})
    qa_pairs.append({"question": "How many karts are to the right of the ego car?", "answer": str(right_count)})
    qa_pairs.append({"question": "How many karts are in front of the ego car?",     "answer": str(front_count)})
    qa_pairs.append({"question": "How many karts are behind the ego car?",          "answer": str(back_count)})

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_all(
    data_dir: str = "../data/train",
    output_file: str = "../data/train/train_qa_pairs.json",
):
    """
    Walk every *_info.json in data_dir, generate QA pairs for every view,
    attach the relative image_file path, and write one combined JSON file.

    Usage:
        python -m homework.generate_qa generate_all \
            --data_dir ../data/train \
            --output_file ../data/train/train_qa_pairs.json
    """
    data_path = Path(data_dir)
    info_files = sorted(data_path.glob("*_info.json"))

    if not info_files:
        print(f"No *_info.json files found in {data_dir}")
        return

    all_qa: list[dict] = []

    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")

        # Discover how many views exist for this frame
        image_files = sorted(info_file.parent.glob(f"{base_name}_*_im.jpg"))

        for image_file in image_files:
            _, view_index = extract_frame_info(str(image_file))

            try:
                qa_pairs = generate_qa_pairs(str(info_file), view_index)
            except Exception as e:
                print(f"Skipping {info_file} view {view_index}: {e}")
                continue

            # Store path relative to the data root (one level up from data_dir)
            rel_image = image_file.relative_to(data_path.parent)

            for qa in qa_pairs:
                all_qa.append({
                    "question":   qa["question"],
                    "answer":     qa["answer"],
                    "image_file": str(rel_image),
                })

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_qa, f, indent=2)

    print(f"Wrote {len(all_qa)} QA pairs to {output_path}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python -m homework.generate_qa check --info_file ../data/valid/00000_info.json --view_index 0

Generate the full training dataset:
   python -m homework.generate_qa generate_all \
       --data_dir ../data/train \
       --output_file ../data/train/train_qa_pairs.json
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate_all": generate_all})


if __name__ == "__main__":
    main()