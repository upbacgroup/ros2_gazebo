#!/usr/bin/env python3
import xml.etree.ElementTree as ET

def extract_wall_names(world_file_path):
    tree = ET.parse(world_file_path)
    root = tree.getroot()

    wall_names = []

    for model in root.findall(".//model"):
        model_name = model.get("name")
        for link in model.findall(".//link"):
            link_name = link.get("name")
            if link_name.startswith("Wall_"):
                wall_names.append(link_name)

    return wall_names

# Replace 'your_world_file.world' with the actual path to your .world file
world_file_path = '/home/ecem/Documents/drone_ws/src/air_drone/world/env_and_balls.world'
wall_names = extract_wall_names(world_file_path)

# print("Wall Names:")
# for name in wall_names:
#     print(name)

