import subprocess
import os

def build():
    """Build the Docker image"""
    try:
        subprocess.run([
            "docker", "build",
            "-t", "diffcheck:latest",
            "."
        ], check=True)
        print("Docker image built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building Docker image: {e}")
        return False

def run():
    """Run the Docker container"""
    try:
        subprocess.run([
            "docker", "run",
            "-p", "5000:5000",
            "diffcheck:latest"
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running Docker container: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dagger.py [build|run]")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "build":
        success = build()
    elif command == "run":
        success = run()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: build, run")
        success = False
        
    sys.exit(0 if success else 1)
