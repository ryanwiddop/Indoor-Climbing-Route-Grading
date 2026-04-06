import hashlib, os, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else print("Please provide a directory path as an argument.")
image_hashes = {}

def get_image_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

print(f"Scanning directory: {PATH}")
for filename in os.listdir(PATH):
    file_path = os.path.join(PATH, filename)
    hash_value = get_image_hash(file_path)
    if hash_value in image_hashes:
        image_hashes[hash_value].append(filename)
    else:
        image_hashes[hash_value] = [filename]

duplicates_found = False
for hash_value, filenames in image_hashes.items():
    if len(filenames) > 1:
        duplicates_found = True
        print(f"Duplicate images found: {', '.join(filenames)}")

if not duplicates_found:
    print("No duplicate images found.")
