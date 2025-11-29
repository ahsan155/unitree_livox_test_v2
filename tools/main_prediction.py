# prediction_server.py  (run from your python3.10 venv)
import socket
import json
import threading
import numpy as np
import time

# import your PredictionWorker implementation and model building code
# adjust import path as needed:
from pred_worker import PredictionWorker
# and code to create model or reuse existing logic:
# e.g., from tools.vis_ros_update_2 import build_model_fn (not real—adapt to repo)

import os
from typing import Sequence
from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
from human_scene_transformer.model import model as hst_model
from human_scene_transformer.model import model_params
from human_scene_transformer.pedestrians import dataset_params
from human_scene_transformer.pedestrians import input_fn

HOST = '127.0.0.1'
PORT = 6000

# newline-delimited JSON
def recv_lines(conn, on_msg):
    buf = b""
    while True:
        data = conn.recv(4096)
        if not data:
            break
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                obj = json.loads(line.decode('utf-8'))
            except Exception as e:
                print("Bad json line:", e)
                continue
            on_msg(conn, obj)

def send_json(conn, obj):
    conn.sendall((json.dumps(obj) + "\n").encode('utf-8'))

def handle_client(conn, address, worker):
    print("Client connected:", address)
    def on_msg(conn, obj):
        """
        obj is the parsed JSON object received from the bridge.
        Expected obj format (payload):
          {"tids": [1,2,...], "trajectories": [
                [[t,x,y,yaw], [t,x,y,yaw], ...],   # traj for id1
                [[t,x,y,yaw], [t,x,y,yaw], ...],   # traj for id2
                ...
          ]}
        """
        payload = obj.get("payload", obj)  # support either wrapping or direct payload
        tids = payload.get("tids")
        trajs = payload.get("trajectories")

        # Basic validation
        if not tids or not trajs:
            print("Invalid payload - missing tids or trajectories:", payload)
            return

        if len(tids) != len(trajs):
            print("Mismatched lengths: len(tids)={}, len(trajs)={}".format(len(tids), len(trajs)))
            return

        # Convert each trajectory to a numpy array of shape (T, D)
        traj_arrays = []
        for idx, traj in enumerate(trajs):
            # traj should be a list of [t, x, y, yaw] or [t,x,y] entries
            try:
                arr = np.array(traj, dtype=np.float32)
            except Exception as e:
                print(f"Failed to convert trajectory {idx} to numpy array: {e}; traj={traj}")
                arr = None

            if arr is None or arr.size == 0:
                # If empty, create empty shape (0, D)
                arr = np.zeros((0, 4), dtype=np.float32)
            elif arr.ndim == 1:
                # Single point arrived as a flat list [t,x,y,yaw], convert to (1,D)
                arr = arr.reshape(1, -1)
            # now arr.shape is (T, D)
            traj_arrays.append(arr)

        # Add the batch task (worker expects list of ids and list of trajectories)
        # Example: worker.add_task([id1, id2], [arr1, arr2])
        worker.add_task(tids, traj_arrays)

        # Wait for predictions for all ids with a single overall timeout
        timeout_s = 2.0         # total allowed time to gather all predictions
        poll_interval = 0.01
        deadline = time.time() + timeout_s
        results = {}
        remaining = set(tids)

        while remaining and time.time() < deadline:
            # Poll each remaining track_id
            for tid in list(remaining):
                res = worker.get_result(tid)
                if res is not None:
                    # Ensure result is serializable (convert numpy to list)
                    results[tid] = np.array(res).tolist()
                    remaining.remove(tid)
            if remaining:
                time.sleep(poll_interval)

        # For any track_ids still missing, return empty list (or last-known, depending on logic)
        predictions = []
        for tid in tids:
            pred_traj = results.get(tid, [])
            predictions.append({"track_id": int(tid), "trajectory": pred_traj})

        # Send back predictions as a single JSON message
        send_json(conn, {"type": "prediction", "payload": {"predictions": predictions}})


    try:
        recv_lines(conn, on_msg)
    except Exception as e:
        print("Client handler error:", e)
    finally:
        conn.close()
        print("Client disconnected:", address)

def main():
    # Build your model & PredictionWorker here.
    # Example (adapt to your repository):
    # model_p = ...; model = ...; worker = PredictionWorker(model)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


    logging.set_verbosity(logging.ERROR)
    gin.parse_config_files_and_bindings(
        [os.path.join('./checkpoints', 'params', 'operative_config.gin')],
        None,
        skip_unknown=True)
    print('Actual gin config used:')
    print(gin.config_str())
    
    
    d_params = dataset_params.PedestriansDatasetParams(
        num_agents=None,
        eval_config='test'
        )

    dataset = input_fn.load_dataset(
        d_params,
        split='test',
        augment=False,
        shuffle=False,
        repeat=False,
        deterministic=True,
    )
    
    model_p = model_params.ModelParams()
    model = hst_model.HumanTrajectorySceneTransformer(model_p)
    _, _ = model(next(iter(dataset.batch(1))), training=False)

    checkpoint_path = './checkpoints/ckpts/ckpt-20'
    checkpoint_mngr = tf.train.Checkpoint(model=model)
    checkpoint_mngr.restore(checkpoint_path).assert_existing_objects_matched()
    #status = checkpoint_mngr.restore(checkpoint_path)
    #status.expect_partial()       
    logging.info('Restored checkpoint: %s', checkpoint_path)
    worker = PredictionWorker(model)  # modify constructor if needed

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    print("Prediction server listening on", HOST, PORT)
    try:
        while True:
            conn, addr = s.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr, worker), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        s.close()

if __name__ == "__main__":
    main()

