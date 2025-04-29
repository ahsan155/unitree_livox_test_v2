# Parameters
start_frame = 780
end_frame = 1500
frame_step = 10  # Frame interval
num_steps = (end_frame - start_frame) // frame_step

# Step size so x does not exceed 30
dx = 0.4  # meters per step

# Agent definitions
agents = {
    2: {"start": (0, 2), "dx": dx, "dy": 0},   # Agent 2 moves right
    3: {"start": (0, 5), "dx": dx, "dy": 0}    # Agent 3 moves right
}

output_file = "./synthetic_tracking_data_within_range.txt"

with open(output_file, "w") as f:
    for frame in range(start_frame, end_frame + 1, frame_step):
        for agent_id, cfg in agents.items():
            step_num = (frame - start_frame) // frame_step
            x = cfg["start"][0] + cfg["dx"] * step_num
            y = cfg["start"][1] + cfg["dy"] * step_num
            line = f"{frame}\t{agent_id}\t{x:.2f}\t{y:.2f}\n"
            f.write(line)

print(f"Data saved to {output_file}")
