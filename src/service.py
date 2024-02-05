#!/usr/bin/env python3

import rospy
import rosservice
import os

import tf
import re
#import transformations
from std_srvs.srv import SetBool
from air_drone.srv import ChangeWorldColor, ChangeWorldColorResponse

class ChangeWorldColorService():

    def __init__(self):
        self.change_color_service = rospy.Service('change_world_color', ChangeWorldColor, self.handle_set_material_color)

    def handle_set_material_color(self, req):
        
        world_file = req.world_file
        name = req.name
        ambient = req.ambient

        with open(world_file, 'r') as f:
            world_contents = f.read()
        
        # Print the world_contents for debugging
        # print("World Contents:\n", world_contents)

        # Update the ambient value for the specified target (target_1)
        pattern = r'<visual name=\'{}\'>(.*?)<ambient>.*?<\/ambient>'.format(name)
        
        # Try to find a match using the pattern
        match = re.search(pattern, world_contents, re.DOTALL)

        if match:
            # Replace the old ambient value numbers with the new ones
            new_ambient_string = f'<ambient>{ambient}</ambient>'
            modified_contents = re.sub(
                r'<ambient>.*?</ambient>',  # Match any existing ambient value
                new_ambient_string,  # Replace with the new ambient value
                world_contents,
                count=1  # Ensure only one replacement
            )

            # Save the modified world file
            with open(world_file, 'w') as f:
                f.write(modified_contents)

            return ChangeWorldColorResponse(success=True)
        else:
            print(f"No match found for {name} in the world file.")
            return ChangeWorldColorResponse(success=False)








        # # Modify the material color
        # for match in re.findall(r'(<material name=".*?")\s*<ambient>\s*(\[.*?\])', world_contents):
        #     if match[1] == name:
        #         world_contents = world_contents.replace(match[0], f'{match[0]} <ambient>{ambient}</ambient>')

        # # Save the modified world file
        # with open(world_file, 'w') as f:
        #     f.write(world_contents)

        # return ChangeWorldColorResponse(success=True)

if __name__ == '__main__':
    rospy.init_node('change_world_color_service_node')
    change_world_color = ChangeWorldColorService()
    rospy.spin()


