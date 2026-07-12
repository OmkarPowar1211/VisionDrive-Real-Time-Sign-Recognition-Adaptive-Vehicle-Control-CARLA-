import carla
import numpy as np
import cv2
import time
import atexit
from ultralytics import YOLO

# Load your custom YOLOv5 model
model_path = r'C:\Users\athar\Downloads\TH_OWL\AUV\Team7_Carla\yolov5\runs\train\carla_traffic_light12\weights\best.pt'
model = YOLO(model_path)

# Connect to CARLA simulator
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Find existing vehicle or spawn one
vehicle = None
for actor in world.get_actors():
    if 'vehicle' in actor.type_id:
        vehicle = actor
        print(f"Found existing vehicle with id {vehicle.id}")
        break

if vehicle is None:
    print("No vehicle found. Spawning a new one...")
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]  # choose the first vehicle blueprint
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points found in the map.")
    spawn_point = spawn_points[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned vehicle id: {vehicle.id}")

# Attach an RGB camera to the vehicle
blueprint_library = world.get_blueprint_library()
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
camera_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

def stop_vehicle():
    control = carla.VehicleControl()
    control.throttle = 0.0
    control.brake = 1.0
    vehicle.apply_control(control)

def go_vehicle():
    control = carla.VehicleControl()
    control.throttle = 0.5
    control.brake = 0.0
    vehicle.apply_control(control)

def process_img(image):
    # Convert raw data to numpy array
    img_bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    img_bgr = img_bgra[:, :, :3]

    # Run inference with ultralytics YOLO
    results = model(img_bgr)

    red_detected = False

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                label = model.names[int(cls)]
                if label == 'Traffic_Light_Red':
                    red_detected = True

                # Draw bounding box and label
                cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img_bgr, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Control vehicle based on detection
    if red_detected:
        print("🚦 Red light detected - Stopping vehicle")
        stop_vehicle()
    else:
        go_vehicle()

    # Display image
    cv2.imshow("CARLA YOLOv5 Detection", img_bgr)
    cv2.waitKey(1)

# Clean up actors on exit
def cleanup():
    print("Cleaning up actors...")
    camera.stop()
    camera.destroy()
    if vehicle.is_alive:
        vehicle.destroy()
    cv2.destroyAllWindows()
    print("Cleanup done.")

atexit.register(cleanup)

# Start camera listening
camera.listen(lambda image: process_img(image))

print("Starting YOLOv5 detection on CARLA camera feed. Press Ctrl+C to quit.")

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Exiting...")
