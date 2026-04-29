import mediapipe as mp

print(f"mediapipe version: {mp.__version__}")
print(f"has solutions: {hasattr(mp, 'solutions')}")
print(f"has tasks: {hasattr(mp, 'tasks')}")

if hasattr(mp, 'solutions'):
    print("solutions found!")
    print(f"has hands: {hasattr(mp.solutions, 'hands')}")
else:
    print("INFO: mediapipe does not have 'solutions' (using new Tasks API)")
    
if hasattr(mp, 'tasks'):
    print("tasks found!")
    print(f"Available in tasks: {dir(mp.tasks)}")
    
print(f"\nAll MediaPipe attributes: {dir(mp)}")
