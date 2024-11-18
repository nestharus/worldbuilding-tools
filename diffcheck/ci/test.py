import sys

import anyio
import dagger


async def main():
    """Build the Docker image using Dagger"""
    async with dagger.Connection(dagger.Config(log_output=sys.stderr)) as client:
        # Create Python container
        source = (
            client.container()
            .from_("python:3.12-slim")
            .with_directory("/app", client.host().directory(".", exclude=["ci/"]))
        )

        runner = (source
            .with_workdir("/app")
            .with_exec(["apt-get", "update"])
            .with_exec(["apt-get", "install", "-y", "git"])
            .with_exec(["pip", "install", "pipenv"])
            .with_exec(["pipenv", "lock"])
            .with_exec(["pipenv", "install", "--dev"])
            .with_exec(["pipenv", "run", "setup"])
            .with_exec(["pytest", "tests"])
        )

        out = await runner.stderr()

        print(out)


anyio.run(main)
