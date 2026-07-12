import carla
import math
import random
import time
import queue
import numpy as np
import cv2
import os

# Connect to the CARLA server
client = carla.Client('localhost', 2000)

# Load Town01 and get the world object
world = client.load_world("Town01")
world = client.get_world()

# Get blueprint library (contains definitions for vehicles, sensors, etc.)
bp_lib = world.get_blueprint_library()

# Spawn a vehicle
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

# Spawn a front RGB camera and attach to vehicle
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1280')
camera_bp.set_attribute('image_size_y', '720')
camera_bp.set_attribute('fov', '110')
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Enable autopilot for the vehicle to drive itself
vehicle.set_autopilot(True)

# Enable synchronous mode for deterministic stepping
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Create a queue to receive camera images
image_queue = queue.Queue()
camera.listen(image_queue.put)

#Source: https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
# Build camera intrinsic projection matrix
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

# Convert 3D world location to 2D image coordinates
def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    if point_camera[2] <= 0:
        return None
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

# Get camera parameters
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

# Compute camera projection matrix
K = build_projection_matrix(image_w, image_h, fov)

# Store labeled bounding boxes for static traffic lights
labeled_bbs = []
labeled_bbs += [(bb, "Traffic Light") for bb in world.get_level_bbs(carla.CityObjectLabel.TrafficLight)]
#labeled_bbs += [(bb, "Traffic Sign") for bb in world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)]

# Get dynamic traffic light actors
traffic_lights = world.get_actors().filter('traffic.traffic_light*')
traffic_signs = world.get_actors().filter('traffic.*sign*')

# Helper to convert traffic light state to text
def get_traffic_light_state_name(state):
    if state == carla.TrafficLightState.Red:
        return "Red"
    elif state == carla.TrafficLightState.Yellow:
        return "Yellow"
    elif state == carla.TrafficLightState.Green:
        return "Green"
    else:
        return "Unknown"

# Set sunny weather conditions
sunny_weather = carla.WeatherParameters(
    cloudiness=0.0,
    precipitation=0.0,
    precipitation_deposits=0.0,
    wetness=0.0,
    fog_density=0.0,
    wind_intensity=5.0,
    sun_altitude_angle=70.0,
    sun_azimuth_angle=90.0
)
world.set_weather(sunny_weather)

# Create OpenCV window and output directories
cv2.namedWindow('CARLA RGB', cv2.WINDOW_AUTOSIZE)
output_dir = "D:/E/TH OWl/Autonomous Vehicles/Team 7/Captured Images/output/Images/"
output_dir2 = "D:/E/TH OWl/Autonomous Vehicles/Team 7/Captured Images/output/labels/"
os.makedirs(output_dir, exist_ok=True)

try:
    while True:
        # Advance simulation by one tick
        world.tick()

        # Get next camera image
        image = image_queue.get()
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        bboxes = []  # List to collect YOLO annotations

        # Process each labeled bounding box
        for bb, label in labeled_bbs:
            # Only consider objects in a specific distance range
            if 5.0 < bb.location.distance(vehicle.get_transform().location) < 20.0:
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = bb.location - vehicle.get_transform().location
                if forward_vec.dot(ray) > 1:
                    verts = [v for v in bb.get_world_vertices(carla.Transform())]
                    points_2d = []

                    # Project 3D corners to 2D image points
                    for v in verts:
                        p = get_image_point(v, K, world_2_camera)
                        if p is None:
                            continue
                        if 0 <= p[0] < image_w and 0 <= p[1] < image_h:
                            points_2d.append(p)

                    if len(points_2d) == 0:
                        continue

                    # Compute 2D bounding box in image coordinates
                    points_2d = np.array(points_2d)
                    x_min, y_min = np.min(points_2d, axis=0).astype(int)
                    x_max, y_max = np.max(points_2d, axis=0).astype(int)

                    # Check for dynamic traffic light state
                    traffic_light_state_text = None
                    for tl in traffic_lights:
                        if tl.get_transform().location.distance(bb.location) < 4.0:
                            traffic_light_state_text = get_traffic_light_state_name(tl.get_state())
                            break

                    # If the light is yellow, annotate and save
                    if traffic_light_state_text == "Yellow":
                        label_text = f"Traffic Light Yellow"
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                        cv2.putText(img, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        # Create YOLO annotation (class 2)
                        x_center = ((x_min + x_max) / 2) / image_w
                        y_center = ((y_min + y_max) / 2) / image_h
                        width = (x_max - x_min) / image_w
                        height = (y_max - y_min) / image_h
                        bboxes.append(f"2 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save image and label file if any yellow traffic lights were detected
        if bboxes:
            frame_id = image.frame
            img_path = os.path.join(output_dir, f"frame_{frame_id:04d}.png")
            txt_path = os.path.join(output_dir, f"frame_{frame_id:04d}.txt")

            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(img_path, img_bgr)

            with open(txt_path, 'w') as f:
                f.write("\n".join(bboxes))

        # Display the camera feed
        cv2.imshow('CARLA RGB', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up
    camera.stop()
    vehicle.destroy()
    cv2.destroyAllWindows()
    world.apply_settings(carla.WorldSettings(synchronous_mode=False))
