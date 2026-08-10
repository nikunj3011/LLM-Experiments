import os
import json
import random
import time
import shutil
import urllib.request
import websocket
import config
import traceback

class ComfyClient:
    """Handles communication with the ComfyUI API & WebSocket."""
    def __init__(self, server_url, client_id):
        self.server_url = server_url
        self.client_id = client_id

    def format_time(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{int(h)}h {int(m)}m {s:.1f}s"
        elif m > 0:
            return f"{int(m)}m {s:.1f}s"
        return f"{s:.1f}s"

    def queue_prompt(self, workflow):
        p = {"prompt": workflow, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_url}/prompt", data=data)
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))

    def track_generation_progress(self, ws):
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                msg_type = message.get("type")
                
                if msg_type == "executing":
                    data = message.get("data")
                    if data.get("node") is None and data.get("prompt_id"):
                        print("\n[Success] Generation process finished!")
                        break
                    else:
                        print(f"  Executing Node: {data.get('node')}...", end="\r")
                elif msg_type == "progress":
                    data = message.get("data")
                    print(f"  Step {data['value']}/{data['max']}...", end="\r")


class ImageGenerator:
    """Manages prompt parsing and image generation execution."""
    def __init__(self, client: ComfyClient):
        self.client = client

    def load_prompts_json(self, json_file_path):
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Prompts file not found at: {json_file_path}")
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def get_random_expression(self, categories):
        for cat in categories:
            cat_name = cat.get("category", "").lower()
            if "expressions" in cat_name or "expression" in cat_name:
                items = cat.get("items", [])
                if items:
                    selected = random.choice(items)
                    desc = selected.get("description", "")
                    if desc:
                        return f", {desc}"
        return ""

    def select_positions(self, categories):
        category_quotas = {
            "positions": config.RANDOM_POSITIONS_ITEM,
            "minimal_pos": config.RANDOM_MINIMAL_POSITIONS_ITEM,
            "tease": config.RANDOM_TEASE_POSITIONS_ITEM
        }
        selected = []
        for cat in categories:
            cat_name = cat.get("category", "").lower()
            items = cat.get("items", [])

            for target_cat, quota in category_quotas.items():
                if target_cat in cat_name:
                    if config.RANDOM_IMAGES:
                        sample_count = min(quota, len(items))
                        sampled_items = random.sample(items, sample_count)
                    else:
                        sampled_items = items

                    for item in sampled_items:
                        item_copy = dict(item)
                        item_copy["_source_category"] = target_cat
                        selected.append(item_copy)
                    break
        return selected
    
    def load_combined_prompt(self, json_file_path):
        """
        Parses the JSON structure, iterates through every category,
        selects one random item per category, and builds a combined string.
        Skips positions and expressions so they can be handled separately.
        """
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Prompts file not found at: {json_file_path}")

        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data if isinstance(data, list) else [data]

        combined_descriptions = []

        for cat in categories:
            items = cat.get("items", [])
            cat_name = cat.get("category", "").lower()
            if not items:
                continue
                
            # Skip position and expression categories here
            if "positions" in cat_name or "expressions" in cat_name or "minimal_pos" in cat_name  or "tease" in cat_name or "clothing" in cat_name:
                continue

            selected_item = random.choice(items)
            description = selected_item.get("description", "")

            if description:
                combined_descriptions.append(description)

        descriptions_str = ", ".join(combined_descriptions)
        return f", {descriptions_str} " if descriptions_str else ""

    def run(self):
        if not os.path.exists(config.IMAGE_WORKFLOW_FILE):
            print(f"[Error] Image workflow file not found: {config.IMAGE_WORKFLOW_FILE}")
            return

        with open(config.IMAGE_WORKFLOW_FILE, "r", encoding="utf-8") as f:
            base_workflow = json.load(f)

        categories = self.load_prompts_json(config.PROMPTS_FILE)
        selected_positions = self.select_positions(categories)

        if not selected_positions:
            print("[Warning] No matching positions found to run image generation.")
            return

        total_positions = len(selected_positions)
        total_pairs = sum(
            min(config.RANDOM_CAMERAS, len(p.get("cameras", []))) if config.RANDOM_IMAGES else len(p.get("cameras", []))
            for p in selected_positions
        )
        total_expected_images = total_pairs * config.IMAGES_PER_ITEM
        total_images_completed = 0

        print(f"--- STARTING IMAGE GENERATION ({total_expected_images} images) ---")
        batch_start_time = time.time()
        additional_file_prompt_text = self.load_combined_prompt(config.PROMPTS_FILE)
        for pos_index, pos_obj in enumerate(selected_positions, 1):
            pos_name = pos_obj.get("title", "Unknown Position")
            pos_desc = pos_obj.get("description", "")
            source_cat = pos_obj.get("_source_category", "")
            cameras = pos_obj.get("cameras", [])

            current_men_prompt = "" if source_cat == "minimal_pos" else config.GLOBAL_PROMPT_MEN

            if config.RANDOM_IMAGES:
                sample_size_cam = min(config.RANDOM_CAMERAS, len(cameras))
                selected_cameras = random.sample(cameras, sample_size_cam) if cameras else []
            else:
                selected_cameras = cameras

            print(f"\n📍 POSITION {pos_index}/{total_positions}: {pos_name} [{source_cat}]")

            for cam_index, cam_obj in enumerate(selected_cameras, 1):
                cam_title = cam_obj.get("title", "Unknown Camera")
                cam_desc = cam_obj.get("description", "")
                pos_cam_prompt = f"{pos_name}, {pos_desc}, {cam_desc}".strip(", ")

                for var_index in range(1, config.IMAGES_PER_ITEM + 1):
                    workflow = json.loads(json.dumps(base_workflow))
                    random_expression = self.get_random_expression(categories)
                    random_seed = random.randint(1, 1125899906842624)

                    full_prompt = (
                        config.GIRL_PROMPT
                        + current_men_prompt
                        + config.GLOBAL_PROMPT
                        + pos_cam_prompt
                        + random_expression
                        + additional_file_prompt_text
                        + config.END_GLOBAL_PROMPT
                    )

                    print("=" * 60)
                    print(f"  [FULL PROMPT] '{full_prompt}'")
                    print("  [PROMPT COMPONENTS BREAKDOWN]")
                    print(f"    • girl:                        '{config.GIRL_PROMPT}'")
                    print(f"    • current_men_prompt:          '{current_men_prompt}'")
                    print(f"    • global_prompt:               '{config.GLOBAL_PROMPT}'")
                    print(f"    • pos_cam_prompt:              '{pos_cam_prompt}'")
                    print(f"    • random_expression:           '{random_expression}'")
                    print(f"    • additional_file_prompt_text: '{additional_file_prompt_text}'")
                    print(f"    • end_global_prompt:           '{config.END_GLOBAL_PROMPT}'")
                    print("=" * 60)

                    if config.CLIP_TEXT_NODE_ID in workflow:
                        workflow[config.CLIP_TEXT_NODE_ID]["inputs"]["text"] = full_prompt
                    if config.KSAMPLER_NODE_ID in workflow:
                        workflow[config.KSAMPLER_NODE_ID]["inputs"]["seed"] = random_seed
                    if config.KSAMPLER_NODE_2_ID in workflow:
                        workflow[config.KSAMPLER_NODE_2_ID]["inputs"]["seed"] = random_seed

                    try:
                        ws = websocket.WebSocket()
                        ws.connect(f"ws://{self.client.server_url}/ws?clientId={self.client.client_id}")
                        self.client.queue_prompt(workflow)
                        self.client.track_generation_progress(ws)
                        ws.close()

                        total_images_completed += 1
                    except Exception as e:
                        print(f"  [Error] Generation failed: {e}")

        duration = time.time() - batch_start_time
        print(f"\n🎉 IMAGE GENERATION COMPLETED! ({total_images_completed}/{total_expected_images} images in {self.client.format_time(duration)})\n")


class VideoGenerator:
    """Manages video generation tasks from JSON input files."""
    def __init__(self, client: ComfyClient):
        self.client = client

    def _prepare_image_input(self, original_path):
        """Copies image from any PC location to ComfyUI input directory if needed."""
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Source image not found: {original_path}")

        filename = os.path.basename(original_path)
        destination_path = os.path.join(config.COMFYUI_INPUT_DIR, filename)

        # If file is not already in ComfyUI input directory, copy it over
        if os.path.abspath(original_path) != os.path.abspath(destination_path):
            shutil.copy(original_path, destination_path)
            print(f"  [Info] Copied image to ComfyUI input folder: {filename}")

        # ComfyUI LoadImage node expects just the filename inside its input folder
        return filename
    
    def run(self, tasks_file=config.VIDEO_TASKS_FILE):
        if not os.path.exists(config.VIDEO_WORKFLOW_FILE):
            print(f"[Error] Video workflow file not found: {config.VIDEO_WORKFLOW_FILE}")
            return

        if not os.path.exists(tasks_file):
            print(f"[Error] Video tasks file not found: {tasks_file}")
            return

        with open(config.VIDEO_WORKFLOW_FILE, "r", encoding="utf-8") as f:
            base_workflow = json.load(f)

        with open(tasks_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        print(f"--- STARTING VIDEO GENERATION ({len(tasks)} tasks) ---")
        batch_start = time.time()

        for idx, task in enumerate(tasks, 1):
            title = task.get("title", f"Video Task {idx}")
            img_path = task.get("image_path", "")
            prompt_text = task.get("prompt", "")

            print(f"\n🎬 VIDEO {idx}/{len(tasks)}: {title}")
            print(f"  • Source Image: {img_path}")
            print(f"  • Video Prompt: '{prompt_text}'")

            try:
                img_filename = self._prepare_image_input(img_path)
                workflow = json.loads(json.dumps(base_workflow))
                random_seed = random.randint(1, 1125899906842624)

                # Inject configurations
                if config.VIDEO_IMAGE_LOAD_NODE_ID in workflow:
                    workflow[config.VIDEO_IMAGE_LOAD_NODE_ID]["inputs"]["image"] = img_filename
                
                if config.VIDEO_PROMPT_NODE_ID in workflow:
                    workflow[config.VIDEO_PROMPT_NODE_ID]["inputs"]["prompt"] = prompt_text

                if config.VIDEO_SEED_NODE_ID in workflow:
                    workflow[config.VIDEO_SEED_NODE_ID]["inputs"]["noise_seed"] = random_seed

                ws = websocket.WebSocket()
                ws.connect(f"ws://{self.client.server_url}/ws?clientId={self.client.client_id}")
                self.client.queue_prompt(workflow)
                self.client.track_generation_progress(ws)
                ws.close()

            except Exception as e:
                traceback.print_exc()
                print(f"  [Error] Video generation failed: {e}")

        duration = time.time() - batch_start
        print(f"\n🎉 VIDEO GENERATION COMPLETED! ({len(tasks)} videos in {self.client.format_time(duration)})\n")


import config
# from generators import ComfyClient, ImageGenerator, VideoGenerator

def main():
    # Initialize API Client
    client = ComfyClient(config.COMFYUI_SERVER, config.CLIENT_ID)

    # Initialize Generator Classes
    image_gen = ImageGenerator(client)
    video_gen = VideoGenerator(client)

    # Enable or disable generator execution pipelines as needed
    RUN_IMAGE_GENERATION = True
    RUN_VIDEO_GENERATION = False

    if RUN_IMAGE_GENERATION:
        image_gen.run()

    if RUN_VIDEO_GENERATION:
        video_gen.run()

if __name__ == "__main__":
    main()