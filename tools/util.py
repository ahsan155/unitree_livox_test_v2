# Updated SimpleTracker that stores historical trajectory data.
import numpy as np

class SimpleTracker_old:
    def __init__(self, max_disappeared=5, max_distance=5.0):
        self.tracks = {}  # {track_id: (centroid, box, score, label, disappear_count)}
        self.next_id = 0
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, pred_boxes, pred_scores, pred_labels):
        centroids = pred_boxes[:, :3]  # Shape: (N, 3)
        used_detections = set()  # Track which detections are already assigned

        if not self.tracks:
            # Initialize tracks for first frame
            for i, centroid in enumerate(centroids):
                self.tracks[self.next_id] = (centroid, pred_boxes[i], pred_scores[i], pred_labels[i], 0)
                self.next_id += 1
        else:
            # Step 1: Update existing tracks with closest unmatched detection
            for track_id, (old_centroid, old_box, old_score, old_label, disappear) in list(self.tracks.items()):
                if len(centroids) == 0:
                    self.tracks[track_id] = (old_centroid, old_box, old_score, old_label, disappear + 1)
                    continue

                # Calculate distances to all centroids
                distances = np.linalg.norm(centroids - old_centroid, axis=1)
                sorted_indices = np.argsort(distances)  # Sort by distance

                # Find the closest unmatched detection
                for idx in sorted_indices:
                    if idx not in used_detections and distances[idx] < self.max_distance:
                        self.tracks[track_id] = (centroids[idx], pred_boxes[idx], pred_scores[idx], pred_labels[idx], 0)
                        used_detections.add(idx)
                        break
                else:
                    # No match found within max_distance
                    self.tracks[track_id] = (old_centroid, old_box, old_score, old_label, disappear + 1)

            # Step 2: Add new tracks for unmatched detections
            for i, centroid in enumerate(centroids):
                if i not in used_detections:
                    self.tracks[self.next_id] = (centroid, pred_boxes[i], pred_scores[i], pred_labels[i], 0)
                    self.next_id += 1

            # Step 3: Remove tracks that have disappeared too long
            self.tracks = {k: v for k, v in self.tracks.items() if v[4] < self.max_disappeared}

        return [(track_id, box, score, label) for track_id, (_, box, score, label, _) in self.tracks.items()]


class SimpleTracker:
    def __init__(self, max_disappeared=5, max_distance=10.0):
        # Each track is a dictionary with keys: centroid, box, score, label, disappear, trajectory.
        self.tracks = {}
        self.next_id = 0
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, pred_boxes, pred_scores, pred_labels, current_time):
        """
        pred_boxes: numpy array of shape (N, 7)
        current_time: float timestamp (e.g., seconds)
        """
        centroids = pred_boxes[:, :3]  # (N, 3)
        used_detections = set()

        if not self.tracks:
            # Create new track for each detection
            for i, centroid in enumerate(centroids):
                self.tracks[self.next_id] = {
                    "centroid": centroid,
                    "box": pred_boxes[i],
                    "score": pred_scores[i],
                    "label": pred_labels[i],
                    "disappear": 0,
                    "trajectory": [np.array([current_time, centroid[0], centroid[1], centroid[2]])]
                }
                self.next_id += 1
        else:
            # Update existing tracks
            for track_id, track in list(self.tracks.items()):
                old_centroid = track["centroid"]
                if len(centroids) == 0:
                    track["disappear"] += 1
                    continue

                distances = np.linalg.norm(centroids - old_centroid, axis=1)
                sorted_indices = np.argsort(distances)
                for idx in sorted_indices:
                    if idx not in used_detections and distances[idx] < self.max_distance:
                        # Update track with new detection
                        new_centroid = centroids[idx]
                        track["centroid"] = new_centroid
                        track["box"] = pred_boxes[idx]
                        track["score"] = pred_scores[idx]
                        track["label"] = pred_labels[idx]
                        track["disappear"] = 0
                        # Append the new position with timestamp to the trajectory
                        track["trajectory"].append(np.array([current_time, new_centroid[0], new_centroid[1], new_centroid[2]]))
                        used_detections.add(idx)
                        break
                else:
                    track["disappear"] += 1

            # Add new tracks for unmatched detections
            for i, centroid in enumerate(centroids):
                if i not in used_detections:
                    self.tracks[self.next_id] = {
                        "centroid": centroid,
                        "box": pred_boxes[i],
                        "score": pred_scores[i],
                        "label": pred_labels[i],
                        "disappear": 0,
                        "trajectory": [np.array([current_time, centroid[0], centroid[1], centroid[2]])]
                    }
                    self.next_id += 1

            # Remove tracks that have disappeared too long
            self.tracks = {k: v for k, v in self.tracks.items() if v["disappear"] < self.max_disappeared}

        # Return list of (track_id, box, score, label, trajectory)
        return [(track_id, track["box"], track["score"], track["label"], track["trajectory"]) for track_id, track in self.tracks.items()]
