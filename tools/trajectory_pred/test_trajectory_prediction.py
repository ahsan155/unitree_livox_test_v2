import glob
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from trajectory_prediction import predict_trajectory


def main():
    folder = './plots'
    record_video = False

    # load sample trajectories
    trajectories, _  = load_trajectories()


    for plot_id in trajectories.keys():
        trajectory_sample = trajectories[plot_id]

        plt.figure()

        for t in range(np.shape(trajectory_sample)[0]):
            replay_trajectory = trajectory_sample[0:t+1, :]
            predicted_trajectory = predict_trajectory(replay_trajectory)

            if predicted_trajectory is None:
                continue

            plt.plot(replay_trajectory[:,1], replay_trajectory[:,2])
            plt.plot(predicted_trajectory[:,1], predicted_trajectory[:,2])

            plt.legend(["Recorded Trajectory", "Prediction"])
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title(plot_id)
            plt.xlim([min(trajectory_sample[:,1]) - 2.0, max(trajectory_sample[:,1]) + 2.0 ])
            plt.ylim([min(trajectory_sample[:, 2]) - 2.0, max(trajectory_sample[:, 2]) + 2.0])

            plt.draw()
            plt.pause(0.1)

            if record_video:
                plt.savefig(folder + "/file%02d.png" % t)

            plt.clf()

        if record_video:
            os.chdir(folder)
            subprocess.call([
                'ffmpeg', '-framerate', '10', '-start_number', '11', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
                f'{plot_id}.mp4'
            ])
            for file_name in glob.glob("*.png"):
                os.remove(file_name)
            os.chdir("..")



def load_trajectories():
    trajectories = {}
    trajectories_interp = {}
    dt = 1

    with open('./0000.txt', 'r') as tracking_file:
        for line in tracking_file:
            line = line.rstrip()
            line = line.split()

            frame = line[0]
            object_id = line[1]
            x = line[10]
            y = line[11]
            z = line[12]

            if object_id not in trajectories.keys():
                trajectories[object_id] = {}

            trajectories[object_id][frame] = np.array([x, y, z])

            #print(line)

    for oid, traj in trajectories.items():
        trajectory_array = np.zeros((len(traj), 4))
        n = 0
        for s, pos in traj.items():
            trajectory_array[n, 0] = int(s)
            trajectory_array[n, 1] = pos[0]
            trajectory_array[n, 2] = pos[1]
            trajectory_array[n, 3] = pos[2]
            n += 1

        # interpolate array
        t_samp = trajectory_array[:, 0]
        x_samp = trajectory_array[:, 1]
        y_samp = trajectory_array[:, 2]
        z_samp = trajectory_array[:, 3]

        t_eval = np.linspace(t_samp[0], t_samp[-1], int((t_samp[-1] - t_samp[0])/dt) + 1)
        x_interp = np.interp(t_eval, t_samp, x_samp)
        y_interp = np.interp(t_eval, t_samp, y_samp)
        z_interp = np.interp(t_eval, t_samp, z_samp)

        trajectory_interp = np.concatenate((t_eval[:,None], x_interp[:,None], y_interp[:,None], z_interp[:,None]), 1)

        trajectories_interp[oid] = trajectory_interp



    return trajectories_interp, trajectories



if __name__ == "__main__":
    main()
