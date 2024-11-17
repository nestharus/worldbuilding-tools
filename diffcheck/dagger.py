import sys
import anyio
import dagger


async def build():
    async with dagger.Connection() as client:
        # Get reference to the local project directory
        src = client.host().directory(".", exclude=[
            "dagger.cue",
            ".git",
            ".pytest_cache",
            "__pycache__",
            "*.pyc"
        ])

        # Build container
        ctr = (client.container()
               .from_("python:3.11-slim")
               .with_workdir("/app")
               .with_exec(["pip", "install", "pipenv"])
               .with_directory(".", src)
               .with_exec(["pipenv", "install", "--deploy"])
               .with_exec(["pipenv", "run", "setup"])
               .with_env("FLASK_ENV", "production")
               .with_exposed_port(5000)
               .with_entrypoint(["pipenv", "run", "start"]))

        # Export the container
        await ctr.publish("text-diff-app:latest")


async def run():
    async with dagger.Connection() as client:
        # Run the container
        ctr = (client.container()
               .from_("text-diff-app:latest")
               .with_exposed_port(5000))
        
        await ctr.with_exec(["pipenv", "run", "start"]).sync()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python dagger.py [build|run]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "build":
        await build()
    elif command == "run":
        await run()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    anyio.run(main)
