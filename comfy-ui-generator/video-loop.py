import os
import sys
import subprocess
import time

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {s:.1f}s"
    elif m > 0:
        return f"{int(m)}m {s:.1f}s"
    return f"{s:.1f}s"

def loop_video(input_path, loop_count, output_path=None):
    """
    Loops a video N times using FFmpeg stream copying (lossless and fast).
    
    :param input_path: Path to the source video file.
    :param loop_count: Number of times to loop the video (e.g., 3 means 3 total plays).
    :param output_path: Optional custom output path. If None, auto-generates filename.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: Input video file not found at '{input_path}'")
        return

    if loop_count <= 0:
        print("❌ Error: Loop count must be greater than 0.")
        return

    # Auto-generate output file path if not provided
    if not output_path:
        dir_name, full_filename = os.path.split(input_path)
        filename, ext = os.path.splitext(full_filename)
        output_path = os.path.join(dir_name, f"{filename}_looped_x{loop_count}{ext}")

    print("=" * 60)
    print(f"🎬 LOOPING VIDEO: {os.path.basename(input_path)}")
    print(f"🔁 Repeats:       {loop_count} times")
    print(f"📁 Output Target: {output_path}")
    print("=" * 60)

    start_time = time.time()

    # Build FFmpeg command
    # -stream_loop N-1 repeats the input stream N-1 additional times (total N plays)
    # -c copy avoids re-encoding, making the process virtually instantaneous
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", str(loop_count - 1),
        "-i", input_path,
        "-c", "copy",
        output_path
    ]

    try:
        print("⚡ Processing...")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Finished successfully in: {format_time(elapsed)}")
        print(f"💾 Saved to: {os.path.abspath(output_path)}\n")

    except FileNotFoundError:
        print("\n❌ Error: FFmpeg is not installed or not found on your system PATH.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during FFmpeg processing: {e}")

if __name__ == "__main__":
    # ==========================================
    # CONFIGURATION / USAGE
    # ==========================================
    
    # 1. Path to your input video
    INPUT_VIDEO = r"D:\Comfy-Desktop\ComfyUI-Shared\output\video\ComfyUI_00077_.mp4"
    
    # 2. Number of times you want the video to play
    LOOPS = 3

    # Run loop function
    loop_video(input_path=INPUT_VIDEO, loop_count=LOOPS)