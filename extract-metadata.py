from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_all_image_metadata(image_path):
    metadata = {}
    
    try:
        with Image.open(image_path) as img:
            # 1. Basic Image Attributes
            metadata["Basic Info"] = {
                "Format": img.format,
                "Mode": img.mode,
                "Dimensions": f"{img.width}x{img.height}"
            }
            
            # 2. Extract PNG / Embedded Text Chunks (where "parameters" lives)
            # This captures Stable Diffusion prompts, generation settings, software info, etc.
            text_metadata = {}
            if hasattr(img, 'info') and img.info:
                for key, val in img.info.items():
                    # Skip binary blobs like exif or icc_profile for clean printing
                    if isinstance(val, (bytes, bytearray)):
                        continue
                    text_metadata[key] = val
            
            metadata["PNG / Text Metadata (img.info)"] = text_metadata if text_metadata else "None"

            # 3. Extract Traditional EXIF Data (if present)
            exif_dict = {}
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if tag_name == "GPSInfo":
                        gps_data = {GPSTAGS.get(k, k): v for k, v in value.items()}
                        exif_dict["GPSInfo"] = gps_data
                    elif not isinstance(value, (bytes, bytearray)):
                        exif_dict[tag_name] = value
                metadata["EXIF Metadata"] = exif_dict
            else:
                metadata["EXIF Metadata"] = "None"

    except Exception as e:
        metadata["Error"] = str(e)
        
    return metadata


if __name__ == "__main__":
    # Remember to use r"..." raw string for Windows paths
    file_path = r"C:\Users\nikun\Downloads\images\a.png"
    
    info = get_all_image_metadata(file_path)
    
    print("=" * 60)
    for section, content in info.items():
        print(f"\n[ {section} ]")
        if isinstance(content, dict):
            for k, v in content.items():
                print(f"  • {k}:")
                # Format multi-line parameters (like long AI prompts) nicely
                if "\n" in str(v):
                    for line in str(v).splitlines():
                        print(f"      {line}")
                else:
                    print(f"      {v}")
        else:
            print(f"  {content}")
    print("=" * 60)