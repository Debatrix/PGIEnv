#!/usr/bin/env python3
import os
from glob import glob
from multiprocessing import Pool, cpu_count
from functools import partial

from pgibench.envs.pgi_alfred.gen.scripts.path_point_calculator import process_scene as process_scene_wrapper


def process_scene(scene_name='FloorPlan1'):
    """
    Main function: Execute complete five-phase path point generation process.

    Args:
        scene_name: Scene name
    """
    return process_scene_wrapper(scene_name)


if __name__ == "__main__":
    # Get scene list
    scene_names = [
        os.path.basename(x).split('-')[0]
        for x in glob('pgibench/envs/pgi_alfred/gen/layouts/*.npy')
    ]

    n = len(scene_names)

    # Get list of processed scenes
    done_scene_names = [
        os.path.basename(x).split('-')[0] for x in glob('running/*.md')
    ]

    # Filter scenes to process
    todo_scene_names = [s for s in scene_names if s not in done_scene_names]

    print(f"Total scenes: {n}")
    print(f"Processed scenes: {len(done_scene_names)}")
    print(f"Scenes to process: {len(todo_scene_names)}")

    if not todo_scene_names:
        print("All scenes already processed")
        exit(0)

    max_workers = 4

    print(f"Processing in parallel with {max_workers} processes")

    # Assign an X display per scene
    scene_display = [(scene, ) for i, scene in enumerate(todo_scene_names)]

    try:
        # Process in parallel with a process pool
        with Pool(processes=max_workers) as pool:
            # Use starmap to pass multiple arguments
            results = pool.starmap(process_scene_wrapper, scene_display)

        print("=" * 60)
        print(f"All scenes processed! Successfully processed {len(results)} scenes")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted by user, waiting for process pool to shut down...")
    except Exception as e:
        print(f"Error during multiprocessing: {e}", exc_info=True)
