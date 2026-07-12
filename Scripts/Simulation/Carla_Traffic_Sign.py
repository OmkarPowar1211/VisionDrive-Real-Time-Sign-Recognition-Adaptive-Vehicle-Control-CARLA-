import carla
import math
import random
import time
import queue
import numpy as np
import cv2
import os

# Connect to the CARLA simulator
client = carla.Client('localhost', 2000)
# Load Town01 map
world = client.load_world("Town01")
world = client.get_world()
bp_lib = world.get_blueprint_library()

# Spawn the ego vehicle
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
spawn_points = world.get_map().get_spawn_points()
spawn_point = spawn_points[5]
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# Attach an RGB camera to the vehicle
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1280')
camera_bp.set_attribute('image_size_y', '720')
camera_bp.set_attribute('fov', '110')
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Enable autopilot on ego vehicle
vehicle.set_autopilot(True)

# Spawn additional traffic vehicles
num_other_vehicles = 8
other_vehicles = []
for i in range(num_other_vehicles):
    bp = random.choice(bp_lib.filter('vehicle.*'))
    point = random.choice(spawn_points)
    veh = world.try_spawn_actor(bp, point)
    if veh:
        veh.set_autopilot(True)
        other_vehicles.append(veh)

# Prepare spawning pedestrians
from carla import Transform, Location, Rotation
walker_bp = bp_lib.filter('walker.pedestrian.*')
spawn_areas = world.get_map().get_spawn_points()

# Controller blueprint for pedestrians
walker_controller_bp = bp_lib.find('controller.ai.walker')
pedestrians = []
controllers = []

# Spawn pedestrians and start walking
num_pedestrians = 20
for i in range(num_pedestrians):
    bp = random.choice(walker_bp)
    spawn_point = random.choice(spawn_areas)
    ped = world.try_spawn_actor(bp, spawn_point)
    if ped:
        ctrl = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=ped)
        ctrl.start()
        # Send them to a random nearby location
        ctrl.go_to_location(Location(
            x=spawn_point.location.x + random.uniform(-10, 10),
            y=spawn_point.location.y + random.uniform(-10, 10),
            z=spawn_point.location.z))
        ctrl.set_max_speed(1.0 + random.random())  # Random speed between 1-2 m/s
        pedestrians.append(ped)
        controllers.append(ctrl)

# Enable synchronous simulation mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Create a queue to receive images
image_queue = queue.Queue()
camera.listen(image_queue.put)


#Source: https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
# Camera projection matrix helper
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

# Convert 3D location to 2D pixel coordinates
def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    # Adjust CARLA coordinate system to image coordinates
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    if point_camera[2] <= 0:
        return None
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

# Define sunny weather parameters
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

# Get camera parameters
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

# Build camera projection matrix
K = build_projection_matrix(image_w, image_h, fov)

# Bounding box edges (not used for 2D but can be used for 3D visualization)
edges = [[0, 1], [1, 3], [3, 2], [2, 0],
         [0, 4], [4, 5], [5, 1], [5, 7],
         [7, 6], [6, 4], [6, 2], [7, 3]]

# Get static bounding boxes for traffic signs
labeled_bbs = []
labeled_bbs += [(bb, "Traffic Sign") for bb in world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)]

# Retrieve dynamic actors for traffic lights and speed signs
traffic_lights = world.get_actors().filter('traffic.traffic_light*')
speed_signs = world.get_actors().filter('traffic.speed_limit.*')

# Helper to get traffic light state name
def get_traffic_light_state_name(state):
    if state == carla.TrafficLightState.Red:
        return "Red"
    elif state == carla.TrafficLightState.Yellow:
        return "Yellow"
    elif state == carla.TrafficLightState.Green:
        return "Green"
    else:
        return "Unknown"

# Helper to identify speed sign type
def get_speed_sign_type(sign_actor):
    tid = sign_actor.type_id
    if tid == "traffic.speed_limit.30":
        return "Speed Limit 30"
    elif tid == "traffic.speed_limit.60":
        return "Speed Limit 60"
    elif tid == "traffic.speed_limit.90":
        return "Speed Limit 90"
    else:
        return "Unknown"

# Create OpenCV window and prepare output directories
cv2.namedWindow('CARLA RGB', cv2.WINDOW_AUTOSIZE)
output_dir = "D:/E/TH OWl/Autonomous Vehicles/Team 7/Captured Images/output/1/Images/"
output_dir2 = "D:/E/TH OWl/Autonomous Vehicles/Team 7/Captured Images/output/1/labels/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)

try:
    while True:
        world.tick()

        # Force all traffic lights to be red briefly
        for light in traffic_lights:
            light.set_state(carla.TrafficLightState.Red)
            light.set_red_time(1.0)
            light.freeze(False)

        # Get latest camera image
        image = image_queue.get()
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        img_bgr_original = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        bboxes = []

        for bb, label in labeled_bbs:
            # Filter bounding boxes by distance
            if 2 < bb.location.distance(vehicle.get_transform().location) < 30.0:
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = bb.location - vehicle.get_transform().location
                if forward_vec.dot(ray) > 1:
                    verts = [v for v in bb.get_world_vertices(carla.Transform())]
                    points_2d = []
                    for v in verts:
                        p = get_image_point(v, K, world_2_camera)
                        if p is None:
                            continue
                        if 0 <= p[0] < image_w and 0 <= p[1] < image_h:
                            points_2d.append(p)

                    if len(points_2d) == 0:
                        continue

                    points_2d = np.array(points_2d)
                    x_min, y_min = np.min(points_2d, axis=0).astype(int)
                    x_max, y_max = np.max(points_2d, axis=0).astype(int)

                    # Match with the closest dynamic speed sign
                    sign_type = "Unknown"
                    for ss in speed_signs:
                        if ss.get_transform().location.distance(bb.location) < 2.0:
                            sign_type = get_speed_sign_type(ss)
                            break

                    # Only process Speed Limit 30 signs
                    if sign_type == "Speed Limit 30":
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(img, "Speed Limit 30", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # YOLO annotation format
                        x_center = ((x_min + x_max) / 2) / image_w
                        y_center = ((y_min + y_max) / 2) / image_h
                        width = (x_max - x_min) / image_w
                        height = (y_max - y_min) / image_h
                        bboxes.append(f"4 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save image and annotations if bounding boxes detected
        if bboxes:
            frame_id = image.frame
            img_path = os.path.join(output_dir, f"frame_Traffic_Sign_30_{frame_id:04d}.png")
            txt_path = os.path.join(output_dir2, f"frame_Traffic_Sign_30_{frame_id:04d}.txt")
            cv2.imwrite(img_path, img_bgr_original)
            with open(txt_path, 'w') as f:
                f.write("\n".join(bboxes))

        cv2.imshow('CARLA RGB', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up
    camera.stop()
    vehicle.destroy()
    cv2.destroyAllWindows()
    world.apply_settings(carla.WorldSettings(synchronous_mode=False))
