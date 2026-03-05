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
}

COLORS = {
    1: (0, 255, 0),  # Green for karts
}

# Original image dimensions for the bounding box coordinates from the engine
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400

def extract_frame_info(image_path: str) -> tuple[int, int]:
    filename = Path(image_path).name
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0

def draw_detections(image_path: str, info_path: str, thickness: int = 2) -> np.ndarray:
    pil_image = Image.open(image_path)
    img_width, img_height = pil_image.size
    draw = ImageDraw.Draw(pil_image)

    with open(info_path) as f:
        info = json.load(f)

    _, view_index = extract_frame_info(image_path)
    frame_detections = info["detections"][view_index]

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        if int(class_id) != 1: continue

        x1_s, y1_s, x2_s, y2_s = x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y
        color = (255, 0, 0) if int(track_id) == 0 else (0, 255, 0)
        draw.rectangle([(x1_s, y1_s), (x2_s, y2_s)], outline=color, width=thickness)

    return np.array(pil_image)

def extract_kart_objects(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    with open(info_path) as f:
        info = json.load(f)
        
    if view_index >= len(info["detections"]): return []

    frame_detections = info["detections"][view_index]
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    karts = []
    for det in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = det
        if int(class_id) != 1: continue

        # Scale and find center
        cx = ((x1 + x2) / 2.0) * scale_x
        cy = ((y1 + y2) / 2.0) * scale_y

        karts.append({
            "name": f"kart_{int(track_id)}",
            "center": (cx, cy),
            "is_ego": (int(track_id) == 0)
        })
    return karts

def extract_track_info(info_path: str) -> str:
    with open(info_path) as f:
        info = json.load(f)
    return info.get("track", "unknown")



def generate_qa_pairs(info_path: str, view_index: int) -> list:
    karts = extract_kart_objects(info_path, view_index)
    track = extract_track_info(info_path)
    qa = []

    ego = next((k for k in karts if k["is_ego"]), None)

    # Core Questions
    if ego:
        qa.append({"question": "What kart is the ego car?", "answer": ego["name"]})
    
    qa.append({"question": "How many karts are there in the scenario?", "answer": str(len(karts))})
    qa.append({"question": "What track is this?", "answer": track})

    # Spatial Reasoning
    if ego:
        ecx, ecy = ego["center"]
        l_cnt = r_cnt = f_cnt = b_cnt = 0

        for k in karts:
            if k["is_ego"]: continue
            kcx, kcy = k["center"]
            
            lr = "left" if kcx < ecx else "right"
            fb = "front" if kcy < ecy else "behind" # Lower Y is further up/front
            
            if lr == "left": l_cnt += 1
            else: r_cnt += 1
            if fb == "front": f_cnt += 1
            else: b_cnt += 1

            qa.append({"question": f"Is {k['name']} to the left or right of the ego car?", "answer": lr})
            qa.append({"question": f"Is {k['name']} in front of or behind the ego car?", "answer": fb})

        qa.append({"question": "How many karts are to the left of the ego car?", "answer": str(l_cnt)})
        qa.append({"question": "How many karts are to the right of the ego car?", "answer": str(r_cnt)})
        qa.append({"question": "How many karts are in front of the ego car?", "answer": str(f_cnt)})
        qa.append({"question": "How many karts are behind the ego car?", "answer": str(b_cnt)})

    return qa

def check(info_file: str, view_index: int):
    info_path = Path(info_file)
    base = info_path.stem.replace("_info", "")
    img_file = list(info_path.parent.glob(f"{base}_{view_index:02d}_im.jpg"))[0]
    
    plt.imshow(draw_detections(str(img_file), info_file))
    plt.title(f"View {view_index}")
    plt.show()

    for pair in generate_qa_pairs(info_file, view_index):
        print(f"Q: {pair['question']} | A: {pair['answer']}")

def generate(data_dir: str = "data/train"):
    path = Path(data_dir)
    split = path.name # e.g., 'train'
    info_files = list(path.glob("*_info.json"))
    
    print(f"Processing {len(info_files)} files in {split}...")
    for info_f in info_files:
        base = info_f.stem.replace("_info", "")
        all_view_qa = []
        for img_f in path.glob(f"{base}_*_im.jpg"):
            _, view_idx = extract_frame_info(str(img_f))
            pairs = generate_qa_pairs(str(info_f), view_idx)
            for p in pairs:
                # FIX: Include the split folder in the path
                p["image_file"] = f"{split}/{img_f.name}"
                all_view_qa.append(p)
        
        with open(path / f"{base}_qa_pairs.json", "w") as f:
            json.dump(all_view_qa, f, indent=4)

if __name__ == "__main__":
    fire.Fire({"generate": generate, "check": check})