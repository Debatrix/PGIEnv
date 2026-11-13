#!/usr/bin/env python3
"""
Virtual X server starter for WSL2 AI2-THOR.
Uses Xvfb instead of Xorg for better stability in WSL2 environment.

Usage:
    python scripts/start_xvfb.py [display_number]
    python scripts/start_xvfb.py 1  # Start on DISPLAY=:1
"""

import subprocess
import sys
import os
import signal
import time


def check_xvfb_installed():
    """Check if Xvfb is installed."""
    try:
        result = subprocess.run(["which", "Xvfb"],
                                capture_output=True,
                                text=True,
                                timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def get_xvfb_process(display_num):
    """Check if Xvfb is already running on the given display."""
    try:
        result = subprocess.run(["ps", "aux"],
                                capture_output=True,
                                text=True,
                                timeout=5)

        for line in result.stdout.split('\n'):
            if f'Xvfb :{display_num}' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]  # Return PID
        return None
    except Exception:
        return None


def start_xvfb(display_num=1, resolution="500x500x24"):
    """
    Start Xvfb on the specified display.

    Args:
        display_num: Display number (e.g., 1 for DISPLAY=:1)
        resolution: Screen resolution (default: 500x500x24)
    """
    display_str = f":{display_num}"

    # Check if Xvfb is installed
    if not check_xvfb_installed():
        print("ERROR: Xvfb is not installed.")
        print("\nInstall with:")
        print("  sudo apt-get install xvfb")
        return False

    # Check if already running
    existing_pid = get_xvfb_process(display_num)
    if existing_pid:
        print(
            f"Xvfb is already running on {display_str} (PID: {existing_pid})")
        response = input("Kill and restart? (y/n): ").lower()
        if response == 'y':
            try:
                subprocess.run(["kill", "-9", existing_pid], check=True)
                time.sleep(0.5)
            except Exception as e:
                print(f"Failed to kill existing process: {e}")
                return False
        else:
            print(f"Using existing Xvfb on {display_str}")
            return True

    # Start Xvfb
    print(f"Starting Xvfb on {display_str} with resolution {resolution}...")
    cmd = [
        "Xvfb",
        display_str,
        "-screen",
        "0",
        resolution,
        "-ac",  # Disable access control
        "+extension",
        "GLX",
        "+render",  # Enable rendering extension
        "+extension",
        "RANDR"
    ]

    try:
        # Start Xvfb in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # Detach from parent process
        )

        # Wait a moment for Xvfb to start
        time.sleep(1)

        # Verify it's running
        if get_xvfb_process(display_num):
            print(
                f"SUCCESS: Xvfb started on {display_str} (PID: {process.pid})")
            print(f"\nSet environment variable:")
            print(f"  export DISPLAY={display_str}")
            return True
        else:
            print("ERROR: Xvfb failed to start")
            return False

    except Exception as e:
        print(f"ERROR: Failed to start Xvfb: {e}")
        return False


def main():
    if len(sys.argv) > 1:
        try:
            display_num = int(sys.argv[1])
        except ValueError:
            print(
                "Invalid display number. Usage: python start_xvfb.py [display_number]"
            )
            return 1
    else:
        display_num = 1

    print("=" * 60)
    print("AI2-THOR Virtual X Server (Xvfb) Starter")
    print("=" * 60)

    success = start_xvfb(display_num)

    if success:
        print("\n" + "=" * 60)
        print("Xvfb is ready. You can now run AI2-THOR environments.")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("Failed to start Xvfb. Check error messages above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
