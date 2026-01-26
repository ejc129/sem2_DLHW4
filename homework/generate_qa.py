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
    with open(info_path) as f:
        info = json.load(f)
    
    # Get detections for this view
    if view_index >= len(info["detections"]):
        return []
    
    frame_detections = info["detections"][view_index]
    
    # Extract kart names list from your specific JSON format
    # Source: info["karts"] = ["gnu", "sara_the_racer", ...]
    kart_names = info.get("karts", [])
    
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    kart_objects = []
    
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)
        
        if class_id != 1:  # Only process karts
            continue
        
        x1_scaled, y1_scaled = x1 * scale_x, y1 * scale_y
        x2_scaled, y2_scaled = x2 * scale_x, y2 * scale_y
        
        # Filter small or out-of-bounds boxes
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue
        
        # NEW MAPPING LOGIC:
        # Use track_id as index for the kart_names list
        if 0 <= track_id < len(kart_names):
            kart_name = kart_names[track_id]
        else:
            kart_name = f"unknown_kart_{track_id}"
        
        kart_objects.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": ((x1_scaled + x2_scaled) / 2, (y1_scaled + y2_scaled) / 2),
            "bbox": (x1_scaled, y1_scaled, x2_scaled, y2_scaled)
        })
    
    # Ego car is track_id 0 (the first name in your 'karts' list)
    for kart in kart_objects:
        kart["is_center_kart"] = (kart["instance_id"] == 0)
    
    return kart_objects

def extract_track_info(info_path: str) -> str:
    with open(info_path) as f:
        info = json.load(f)
    # This matches your JSON key: {"track": "abyss", ...}
    return info.get("track", "unknown")


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
    
    # Extract kart objects
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    
    # Extract track info
    track_name = extract_track_info(info_path)
    
    # Find ego car
    ego_kart = None
    other_karts = []
    for kart in karts:
        if kart["is_center_kart"]:
            ego_kart = kart
        else:
            other_karts.append(kart)
    
    # 1. Ego car question
    if ego_kart:
        qa_pairs.append({
            "question": "What kart is the ego car?",
            "answer": ego_kart["kart_name"]
        })
    
    # 2. Total karts question
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts))
    })
    
    # 3. Track information question
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name
    })
    
    # 4. Relative position questions for each other kart
    if ego_kart:
        ego_x, ego_y = ego_kart["center"]
        
        for kart in other_karts:
            kart_x, kart_y = kart["center"]
            kart_name = kart["kart_name"]
            
            # Left/Right question
            if kart_x < ego_x:
                lr_answer = "left"
            else:
                lr_answer = "right"
            
            qa_pairs.append({
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": lr_answer
            })
            
            # Front/Behind question
            if kart_y < ego_y:
                fb_answer = "in front of"
            else:
                fb_answer = "behind"
            
            qa_pairs.append({
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": fb_answer
            })
            
            # Combined position question
            position = f"{lr_answer} and {fb_answer}"
            qa_pairs.append({
                "question": f"Where is {kart_name} relative to the ego car?",
                "answer": position
            })
    
    # 5. Counting questions
    if ego_kart:
        ego_x, ego_y = ego_kart["center"]
        
        left_count = sum(1 for k in other_karts if k["center"][0] < ego_x)
        right_count = sum(1 for k in other_karts if k["center"][0] >= ego_x)
        front_count = sum(1 for k in other_karts if k["center"][1] < ego_y)
        behind_count = sum(1 for k in other_karts if k["center"][1] >= ego_y)
        
        qa_pairs.append({
            "question": "How many karts are to the left of the ego car?",
            "answer": str(left_count)
        })
        
        qa_pairs.append({
            "question": "How many karts are to the right of the ego car?",
            "answer": str(right_count)
        })
        
        qa_pairs.append({
            "question": "How many karts are in front of the ego car?",
            "answer": str(front_count)
        })
        
        qa_pairs.append({
            "question": "How many karts are behind the ego car?",
            "answer": str(behind_count)
        })
    
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


def generate_all_qa_pairs(data_dir: str = "data/train", output_file: str = None):
    """
    Generate QA pairs for all info files in a directory.
    
    Args:
        data_dir: Directory containing info.json files
        output_file: Optional output file path. If None, saves to {data_dir}/balanced_qa_pairs.json
    """
    import time
    from tqdm import tqdm
    
    start_time = time.time()
    data_path = Path(data_dir)
    
    if output_file is None:
        output_file = data_path / "balanced_qa_pairs.json"
    else:
        output_file = Path(output_file)
    
    print("=" * 80)
    print("QA PAIR GENERATION - STARTING")
    print("=" * 80)
    
    all_qa_pairs = []
    
    # Find all info files
    print("\n[1/4] Scanning directory for info files...")
    info_files = list(data_path.glob("*_info.json"))
    print(f"✓ Found {len(info_files)} info files in {data_dir}")
    
    if len(info_files) == 0:
        print("⚠ WARNING: No info files found! Check the directory path.")
        return
    
    # Count total images for progress tracking
    print("\n[2/4] Counting total images...")
    total_images = 0
    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        image_files = list(data_path.glob(f"{base_name}_*_im.jpg"))
        total_images += len(image_files)
    print(f"✓ Found {total_images} total images to process")
    
    # Process all files with progress bar
    print("\n[3/4] Generating QA pairs...")
    processed_images = 0
    failed_images = 0
    
    with tqdm(total=total_images, desc="Processing images", unit="img") as pbar:
        for info_idx, info_file in enumerate(info_files, 1):
            base_name = info_file.stem.replace("_info", "")
            
            # Find all image files for this info file
            image_files = list(data_path.glob(f"{base_name}_*_im.jpg"))
            
            for image_file in image_files:
                try:
                    _, view_index = extract_frame_info(str(image_file))
                    
                    # Generate QA pairs for this view
                    qa_pairs = generate_qa_pairs(str(info_file), view_index)
                    
                    # Add image_file to each QA pair
                    relative_image_path = f"train/{image_file.name}"
                    
                    for qa in qa_pairs:
                        qa["image_file"] = relative_image_path
                        all_qa_pairs.append(qa)
                    
                    processed_images += 1
                    
                    # Update progress bar with current stats
                    pbar.set_postfix({
                        'QA pairs': len(all_qa_pairs),
                        'Failed': failed_images
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    failed_images += 1
                    pbar.set_postfix({
                        'QA pairs': len(all_qa_pairs),
                        'Failed': failed_images
                    })
                    pbar.update(1)
                    tqdm.write(f"⚠ Error processing {image_file.name}: {str(e)}")
    
    # Save to JSON
    print(f"\n[4/4] Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    elapsed_time = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 80)
    print("QA PAIR GENERATION - COMPLETED")
    print("=" * 80)
    print(f"✓ Total images processed: {processed_images}/{total_images}")
    print(f"✓ Total QA pairs generated: {len(all_qa_pairs)}")
    print(f"✓ Average QA pairs per image: {len(all_qa_pairs)/max(processed_images, 1):.1f}")
    if failed_images > 0:
        print(f"⚠ Failed images: {failed_images}")
    print(f"✓ Output file: {output_file}")
    print(f"✓ Time elapsed: {elapsed_time:.2f} seconds")
    print(f"✓ Processing speed: {processed_images/elapsed_time:.1f} images/sec")
    print("=" * 80)


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate": generate_all_qa_pairs
    })


if __name__ == "__main__":
    main()