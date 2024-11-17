import sys
import anyio
import dagger

async def build():
    """Build the Docker image using Dagger"""
    async with dagger.Connection(dagger.Config(log_output=sys.stderr)) as client:
        # Get reference to the local project
        src = client.host().directory(".")
        
        # Create Python container
        python = (
            client.container()
            .from_("python:3.11-slim")
            .with_workdir("/app")
            .with_exec(["pip", "install", "pipenv"])
            .with_directory("/app", src)
            .with_exec(["pipenv", "install", "--deploy"])
            .with_exec(["pipenv", "run", "setup"])
            .with_env_variable("FLASK_ENV", "production")
            .with_exposed_port(5000)
            .with_entrypoint(["pipenv", "run", "start"])
        )

        # Export the container
        await python.publish("difwordcount:latest")
        print("Docker image built successfully")
        return True

async def run():
    """Run the Docker container"""
    try:
        import subprocess
        subprocess.run([
            "docker", "run",
            "-p", "5000:5000",
            "difwordcount:latest"
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running Docker container: {e}")
        return False
