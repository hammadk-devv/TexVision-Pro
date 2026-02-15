
import os
import sys
import glob

# Try to import tensorboard
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("TensorBoard not installed or not accessible via python script.")

def get_latest_run_dir(run_dir):
    # In this project, runs are files in 'runs/' directory directly
    # or subdirectories?
    # Based on list_dir earlier: runs/events.out.tfevents...
    return run_dir

def read_progress(log_dir):
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print("No event files found.")
        return

    # sort by modification time
    latest_event = max(event_files, key=os.path.getmtime)
    print(f"Reading event file: {latest_event}")

    if HAS_TB:
        ea = EventAccumulator(latest_event)
        ea.Reload()
        
        # Check available tags
        tags = ea.Tags()
        # print(tags)
        
        if 'scalars' in tags:
            scalars = tags['scalars']
            if 'Loss/train' in scalars:
                events = ea.Scalars('Loss/train')
                if events:
                    last_event = events[-1]
                    print(f"Last Logged Epoch: {last_event.step}")
                    print(f"Last Train Loss: {last_event.value:.4f}")
            
            if 'Accuracy/train' in scalars:
                events = ea.Scalars('Accuracy/train')
                if events:
                    print(f"Last Train Accuracy: {events[-1].value:.4f}")
            
            if 'Accuracy/val' in scalars:
                events = ea.Scalars('Accuracy/val')
                if events:
                    print(f"Last Val Accuracy: {events[-1].value:.4f}")

    else:
        print("Cannot read details without tensorboard package. Checking file metadata.")
        print(f"File size: {os.path.getsize(latest_event)} bytes")

if __name__ == "__main__":
    read_progress('runs')
