import argparse
import threading
import time
from queue import Queue

import cv2
from flask import Flask, Response, render_template
from ultralytics import YOLO

from utils import get_local_ip, FPSCounter, VideoWriterOptional

app = Flask(__name__)

# Shared objects
frame_queue = Queue(maxsize=4)   # raw frames from capture
annotated_queue = Queue(maxsize=4)  # annotated JPEG bytes ready to stream
stop_event = threading.Event()


def capture_loop(source, width, height):
    cap = cv2.VideoCapture(source)
    # Try to set resolution if requested
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            if not frame_queue.full():
                frame_queue.put(frame, timeout=0.01)
        except Exception:
            pass
        time.sleep(0)  # yield

    cap.release()


def inference_loop(model, show_fps=True, save_path=None):
    writer = VideoWriterOptional(save_path) if save_path else None
    fps_counter = FPSCounter()

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except Exception:
            continue

        # Run inference
        results = model(frame, verbose=False)  # returns Results
        annotated = results[0].plot()  # returns np.ndarray

        if show_fps:
            fps = fps_counter.tick()
            cv2.putText(annotated, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Optionally save
        if writer:
            writer.write(annotated)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated)
        if ret:
            jpg_bytes = buffer.tobytes()
            try:
                if not annotated_queue.full():
                    annotated_queue.put(jpg_bytes, timeout=0.01)
            except Exception:
                pass

    if writer:
        writer.release()


@app.route('/')
def index():
    return render_template('index.html', ip=get_local_ip())


def generate_mjpeg():
    while not stop_event.is_set():
        try:
            jpg = annotated_queue.get(timeout=1)
        except Exception:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def main(args):
    # Load model (cpu or gpu automatically handled by ultralytics/Yolov8 if torch.cuda available)
    print('Loading model...')
    model = YOLO(args.model)

    # Start threads
    cap_thread = threading.Thread(target=capture_loop, args=(args.source, args.width, args.height), daemon=True)
    inf_thread = threading.Thread(target=inference_loop, args=(model, True, args.save), daemon=True)

    cap_thread.start()
    inf_thread.start()

    try:
        # start flask (blocking) - accessible from phone at http://<PC_IP>:5000
        app.run(host='0.0.0.0', port=args.port, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap_thread.join(timeout=1)
        inf_thread.join(timeout=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0',
                        help='Video source. 0 for webcam, or path to file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path or name of YOLOv8 model')
    parser.add_argument('--port', type=int, default=5000, help='Flask port')
    parser.add_argument('--width', type=int, default=None, help='Optional capture width')
    parser.add_argument('--height', type=int, default=None, help='Optional capture height')
    parser.add_argument('--save', type=str, default=None, help='Optional path to save annotated video')
    args = parser.parse_args()

    # allow passing integer 0
    try:
        if args.source.isdigit():
            args.source = int(args.source)
    except Exception:
        pass

    main(args)