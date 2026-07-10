# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion-planning Step 6b: prove trajectory files saved by step6a_save_trajectory.py can be
loaded and understood correctly, completely separately from planning or robot simulation.

Pure data-loading test -- no cuRobo, no Isaac Sim, no GPU. Just reads the JSON file
step6a wrote (joint_names, position, velocity, interpolation_dt -- see step6a's docstring for why
this JSON exists alongside the .usd: the .usd is visual-only, this JSON is the actual
programmatically-reloadable trajectory data) and prints exactly what's needed to confirm it
matches what step6a printed at save time.

.. code-block:: bash

    # load the most recently saved trajectory
    python3 scripts/motion_planning/step6b_load_trajectory.py

    # load a specific file
    python3 scripts/motion_planning/step6b_load_trajectory.py --file scripts/motion_planning/trajectories/step6a_trajectory_20260710_161616.json

"""

import argparse
import glob
import json
import os
import sys

TRAJECTORY_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")


def find_most_recent_trajectory(directory: str) -> str:
    """Return the most recently modified .json trajectory file in directory."""
    json_files = glob.glob(os.path.join(directory, "*.json"))
    if not json_files:
        print(f"[ERROR] No .json trajectory files found in {directory}. Run step6a_save_trajectory.py first.")
        sys.exit(1)
    return max(json_files, key=os.path.getmtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6b: load and print a saved trajectory JSON.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a trajectory .json file. Defaults to the most recently saved file in "
        f"{TRAJECTORY_SAVE_DIR}.",
    )
    args = parser.parse_args()

    file_path = args.file if args.file is not None else find_most_recent_trajectory(TRAJECTORY_SAVE_DIR)
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        print(f"[ERROR] Trajectory file does not exist: {file_path}")
        sys.exit(1)

    print(f"[INFO] Loading trajectory from: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)

    joint_names = data["joint_names"]
    position = data["position"]
    interpolation_dt = data["interpolation_dt"]
    n_waypoints = len(position)

    print(f"\n[LOAD] joint_names: {joint_names}")
    print(f"[LOAD] Number of waypoints: {n_waypoints}")
    print(f"[LOAD] First waypoint position: {position[0]}")
    print(f"[LOAD] Last waypoint position:  {position[-1]}")
    print(f"[LOAD] interpolation_dt: {interpolation_dt}")

    print(
        "\n[INFO] Compare the five lines above against step6a's terminal output for this same "
        "file -- [PLAN] SUCCESS's waypoint count, [SAVE] Waypoints saved, and the JointState "
        "printed at [STEP 4] -- to confirm the round-trip is exact."
    )


if __name__ == "__main__":
    main()
