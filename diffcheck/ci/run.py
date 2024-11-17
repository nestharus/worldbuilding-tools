import sys

import anyio
import dagger


async def main():
    """Build the Docker image using Dagger"""
    async with dagger.Connection(dagger.Config(log_output=sys.stderr)) as client:
        sys.stdout.reconfigure(encoding='utf-8')

        environment = (
            client.container()
            .from_("python:3.12-slim")
            .with_exec(["apt-get", "update"])
            .with_exec(["apt-get", "install", "-y", "git"])
            .with_exec(["pip", "install", "pipenv"])
            .with_exec(["pip", "install", "dagger-io"])
        )

        setup = (environment
            .with_directory("/setup", client.host().directory("./setup", exclude=["Pipfile.lock"]))
            .with_workdir("/setup")
            .with_exec(["pipenv", "lock"])
            .with_exec(["pipenv", "install", "--deploy"])
            .with_exec(["pipenv", "run", "setup"])
        )

        source = (
            client.container()
            .from_("python:3.12-slim")
        )

        # Install git for model downloads
        runner = (source
            .with_exec(["apt-get", "update"])
            .with_exec(["apt-get", "install", "-y", "git"])
            .with_exec(["pip", "install", "pipenv"])
            .with_exec(["pip", "install", "dagger-io"])
            .with_workdir("/app")
            .with_directory("/app", client.host().directory(".", exclude=["ci/", "test/"]))
            .with_exec(["pipenv", "lock"])
            .with_exec(["pipenv", "install", "--deploy"])
            .with_exec(["pipenv", "run", "setup"])
            .with_env_variable("FLASK_ENV", "production")
            .with_exposed_port(5000)
            .with_entrypoint(["pipenv", "run", "start"])
        )

        out = await runner.stdout()
        print(out.encode('utf-8', errors='ignore').decode('utf-8'))


anyio.run(main)
