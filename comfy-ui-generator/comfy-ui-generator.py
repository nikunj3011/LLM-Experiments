import config
from generators import ComfyClient, ImageGenerator, VideoGenerator

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