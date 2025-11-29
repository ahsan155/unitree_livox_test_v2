import numpy as np
import threading
import queue
import tensorflow as tf

class PredictionWorker:
    def __init__(self, model):
        self.model = model
        self.task_queue = queue.Queue()
        self.results = {}
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run_worker, daemon=True)
        self.thread.start()
        self.scene_id = tf.constant([b'unitree_tracking_output'], shape=(1,), dtype=tf.string)
        self.scene_timestamp = tf.constant([0], shape=(1,), dtype=tf.int64)

    def _run_worker(self):
        """Background thread that processes prediction tasks."""
        while True:
            task = self.task_queue.get()
            if task is None:  # Sentinel to stop the thread
                break

            track_ids, trajectories = task
            num_agents = len(trajectories)

            if num_agents == 0:
                continue

            # Prepare batched input with correct shape (1, num_agents, 20, 2)
            positions_batch = np.zeros((1, num_agents, 20, 2), dtype=np.float32)
            masks_batch = np.zeros((1, num_agents, 20), dtype=bool)
            
            print('number of agents', num_agents)            
            for i, traj in enumerate(trajectories):
                historical_pos = np.array(traj)[-8:, 1:3]  # Shape (8, 2)
                positions_batch[0, i, :8, :] = historical_pos
                masks_batch[0, i, :8] = True

            # Create complete input batch
            input_batch = {
                'agents/position': tf.convert_to_tensor(positions_batch),
                'agents/mask': tf.convert_to_tensor(masks_batch),
                'scene/id': self.scene_id,
                'scene/timestamp': self.scene_timestamp,
                # Add any other required dummy fields here
            }

            # Run inference
            full_pred, _ = self.model(input_batch, training=False)
            ml_indices = tf.squeeze(tf.math.argmax(full_pred['mixture_logits'], axis=-1))
            preds = full_pred['agents/position'][..., ml_indices, :]  # Shape (1, num_agents, 20, 2)
            
            # We only want future predictions (last 12 timesteps)
            future_preds = preds[0, :, 8:, :]  # Shape (num_agents, 12, 2)

            # Store results
            with self.lock:
                for i, track_id in enumerate(track_ids):
                    self.results[track_id] = future_preds[i].numpy()

    def add_task(self, track_ids, trajectories):
        """Add a new prediction task to the queue."""
        self.task_queue.put((track_ids, trajectories))

    def get_result(self, track_id):
        """Retrieve the latest prediction for a track_id (thread-safe)."""
        with self.lock:
            return self.results.get(track_id, None)

    def stop(self):
        """Stop the worker thread gracefully."""
        self.task_queue.put(None)
        self.thread.join()
