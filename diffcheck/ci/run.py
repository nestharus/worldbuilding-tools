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

        # Setup container to download models
        setup = (environment
            .with_directory("/setup", client.host().directory("./setup", exclude=["Pipfile.lock"]))
            .with_workdir("/setup")
            .with_exec(["pipenv", "lock"])
            .with_exec(["pipenv", "install", "--deploy"])
            .with_exec(["pipenv", "run", "setup"])
        )

        # Create a directory to store models
        models_dir = setup.directory("/root/.cache/tokenizer_models")
        
        # Get all spacy model directories
        spacy_base = "/usr/local/lib/python3.12/site-packages"
        spacy_models = setup.directory(spacy_base, include=["en_core_web*"])

        # Create runtime container 
        source = client.container().from_("python:3.12-slim")

        # Install dependencies and copy models to runtime container
        runner = (source
            .with_exec(["apt-get", "update"])
            .with_exec(["apt-get", "install", "-y", "git"])
            .with_exec(["pip", "install", "pipenv"])
            .with_exec(["pip", "install", "dagger-io"])
            .with_workdir("/app")
            .with_directory("/app", client.host().directory(".", exclude=["ci/", "test/"]))
            # Copy models from setup container
            .with_directory("/root/.cache/tokenizer_models", models_dir)
            .with_directory("/usr/local/lib/python3.12/site-packages", spacy_models)
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
