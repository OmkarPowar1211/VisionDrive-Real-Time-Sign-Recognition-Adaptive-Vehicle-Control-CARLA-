import glob
import os
import sys
import cv2
import numpy as np
import torch
import carla
import time
import traceback
import queue

# ==== Add YOLOv5 Path and Load Model ====
YOLOV5_PATH = r'D:\E\TH OWl\Autonomous Vehicles\Team 7\yolov5-master\yolov5-master'
MODEL_PATH = r'D:\E\TH OWl\Autonomous Vehicles\Team 7\carla_traffic_light17\weights\best.pt'

model = torch.hub.load(YOLOV5_PATH, 'custom', path=MODEL_PATH, source='local')
model.conf = 0.25
model.iou = 0.4
model.classes = None
model.eval()

# ==== Connect to CARLA ====
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town02')
client.reload_world(False)                # reload map keeping world settings

# Set synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.09
world.apply_settings(settings)

# ==== Spawn Ego Vehicle ====
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_points = world.get_map().get_spawn_points()
ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[68])

# ==== Spawn RGB Camera ====
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1280')
camera_bp.set_attribute('image_size_y', '720')
camera_bp.set_attribute('fov', '110')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

# === Class names ===
class_names = [
    'Traffic_Light_Red',
    'Traffic_Light_Green',
    'Traffic_Light_Yellow',
    'Traffic_Sign_30'
]

light_ids = [i for i, n in enumerate(class_names) if 'light' in n.lower()]
speed_sign_ids = [i for i, n in enumerate(class_names) if 'sign' in n.lower()]

# Remove all 60 and 90 speed limit signs
for sign in world.get_actors().filter('traffic.speed_limit.60'):
    sign.destroy()
for sign in world.get_actors().filter('traffic.speed_limit.90'):
    sign.destroy()

def waypoints_navigation(world, client, ego_vehicle):
    map = world.get_map()
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    route = [
        # carla.Location(x=-7.5, y=177.2, z=0,),  # First location
        # carla.Location(x=11.4, y=191.6, z=0), # Second location
        carla.Location(x=23.4, y=191.6, z=0), # Second location
        # carla.Location(x=41.9, y=225.6, z=0), # Third location
        # carla.Location(x=178.5, y=241.0, z=0),   # Fourth location
        # carla.Location(x=193.7, y=204.3, z=0),  # Fifth location
        carla.Location(x=193.7, y=121.1, z=0),  # Fifth location
        carla.Location(x=-7.5, y=177.2, z=0,),  # First location
    ]

    ego_vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.set_path(ego_vehicle, route)
    traffic_manager.ignore_lights_percentage(ego_vehicle, 100)
    traffic_manager.ignore_signs_percentage(ego_vehicle, 100)
    traffic_manager.ignore_vehicles_percentage(ego_vehicle, 100)
    traffic_manager.update_vehicle_lights(ego_vehicle, True)
    traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 0.0)

    print(f"Ego vehicle {ego_vehicle.id} is now following the custom route.")
    return traffic_manager

sunny_weather = carla.WeatherParameters(
    cloudiness=0.0,
    precipitation=0.0,
    precipitation_deposits=0.0,
    wetness=0.0,
    fog_density=0.0,
    wind_intensity=5.0,
    sun_altitude_angle=70.0,
    sun_azimuth_angle=210.0
)

image_queue = queue.Queue()
camera.listen(image_queue.put)
cv2.namedWindow("YOLOv5 Object Detection", cv2.WINDOW_AUTOSIZE)

MIN_BOX_WIDTH_SPEED_SIGNS = 20  # px

try:
    traffic_manager = waypoints_navigation(world, client, ego_vehicle)
    world.set_weather(sunny_weather)

    while True:
        world.tick()

        if image_queue.empty():
            continue

        image = image_queue.get()
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb_img = arr[:, :, :3].copy()

        results = model(rgb_img)
        detections = results.xyxy[0].cpu().numpy()

        # Get all traffic light actors in the world
        all_traffic_lights = world.get_actors().filter("traffic.traffic_light*")

        target_speed_kmh = None

        if detections.size > 0:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                class_id = int(cls)
                label = class_names[class_id]
                print(f"- Detected: {label} ({conf:.2f})")
                
                if class_id in light_ids:
                    # Check distance to nearest traffic light
                    ego_loc = ego_vehicle.get_location()
                    dist_to_closest_light = None

                    # Find the closest traffic light
                    for light in all_traffic_lights:
                        dist = ego_loc.distance(light.get_location())
                        if dist_to_closest_light is None or dist < dist_to_closest_light:
                            dist_to_closest_light = dist

                    # Now decide speed based on label and distance
                    if label == "Traffic_Light_Red":
                        if dist_to_closest_light is not None:
                            if dist_to_closest_light <= 11:
                                print(f"Red light detected at {dist_to_closest_light:.2f} m. Stopping vehicle.")
                                target_speed_kmh = 0
                            else:
                                print(f"Red light detected at {dist_to_closest_light:.2f} m. Slowing down.")
                                target_speed_kmh = 10
                        else:
                            target_speed_kmh = 10
                    elif label == "Traffic_Light_Yellow":
                        target_speed_kmh = 15
                    elif label == "Traffic_Light_Green":
                        target_speed_kmh = 35


                elif label == "Traffic_Sign_30":
                    target_speed_kmh = 25

        if target_speed_kmh is not None:
            target_speed_mps = target_speed_kmh / 3.6
            traffic_manager.vehicle_percentage_speed_difference(
                ego_vehicle,
                100 - ((target_speed_mps / 13.89) * 100)
            )
            print(f"Target speed set to {target_speed_kmh} km/h")

        # Filtered detections for drawing boxes
        filtered_detections = []
        img_width = rgb_img.shape[1]
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            class_id = int(cls)
            width = x2 - x1

            if class_id in speed_sign_ids and width < MIN_BOX_WIDTH_SPEED_SIGNS:
                continue

            if class_id in speed_sign_ids:
                center_x = (x1 + x2) // 2
                if center_x < img_width // 2:
                    continue

            filtered_detections.append((x1, y1, x2, y2, conf, class_id))

        # Draw bounding boxes
        for (x1, y1, x2, y2, conf, class_id) in filtered_detections:
            label = f"{class_names[class_id]} {conf:.2f}"
            if class_id in light_ids:
                cname = class_names[class_id].lower()
                if 'red' in cname:
                    color = (0, 0, 255)
                elif 'yellow' in cname:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(rgb_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
            cv2.putText(rgb_img, f"Speed: {int(target_speed_kmh)} km/h", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("YOLOv5 Object Detection", rgb_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.stop()
    if ego_vehicle is not None:
        ego_vehicle.destroy()
    if camera is not None:
        camera.destroy()

    cv2.destroyAllWindows()
    print("All actors destroyed. Simulation ended.")