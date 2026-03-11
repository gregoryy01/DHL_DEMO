import yaml
import os

def create_data_yaml(project_root):
    # project_root example:
    # C:/Users/lenovo/Documents/GitHub/DHL_DEMO

    data_root = os.path.join(project_root, "data", "raw")
    classes_txt = os.path.join(data_root, "classes.txt")
    data_yaml = os.path.join(data_root, "data.yaml")

    # Read classes.txt
    if not os.path.exists(classes_txt):
        print(f"classes.txt not found at: {classes_txt}")
        return

    with open(classes_txt, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    number_of_classes = len(classes)

    # Create YAML content
    data = {
        "path": data_root.replace("\\", "/"),
        "train": "train/images",
        "val": "validation/images",
        "nc": number_of_classes,
        "names": classes
    }

    # Write YAML file
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"Created config file at: {data_yaml}")
    print("\nFile contents:\n")
    with open(data_yaml, "r", encoding="utf-8") as f:
        print(f.read())


# Set your project root here
project_root = r"C:\Users\lenovo\Documents\GitHub\DHL_DEMO"

create_data_yaml(project_root)