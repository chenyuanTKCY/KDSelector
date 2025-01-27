import os
def find_latest_file(directory):
    
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return None
    
    
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def read_file_as_bytes(file_path):
    
    with open(file_path, "rb") as file:
        return file.read()