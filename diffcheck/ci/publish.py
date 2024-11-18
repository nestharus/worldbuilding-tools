import sys
import anyio
import dagger

async def main():
    """Build the Docker image using Dagger with multi-stage optimization"""
    async with dagger.Connection(dagger.Config(log_output=sys.stderr)) as client:
        builder = (
            client.container()
            .from_("python:3.12-slim")
            .with_exec(["pip", "install", "--no-cache-dir", "pipenv"])
            .with_directory("/setup", client.host().directory("./setup"))
            .with_workdir("/setup")
            .with_exec(["pipenv", "requirements"])
            .with_exec(["sh", "-c", "pipenv requirements > requirements.txt"])
            .with_exec(["pip", "install", "--no-cache-dir", "-r", "requirements.txt"])
            .with_exec(["python", "./setup.py"])
            .with_exec(["pip", "install", "--no-cache-dir", "spacy==3.8.2"])
            .with_exec(["python", "-m", "spacy", "download", "en_core_web_sm"])
        )
        models_dir = builder.directory("/root/.cache/tokenizer_models")
        spacy_model = builder.directory('/usr/local/lib/python3.12/site-packages/en_core_web_sm')
        model_dist = builder.directory('/usr/local/lib/python3.12/site-packages/en_core_web_sm-3.8.0.dist-info')

        app_reqs = (
            client.container()
            .from_("python:3.12-slim")
            .with_exec(["pip", "install", "--no-cache-dir", "pipenv"])
            .with_file("/app/Pipfile", client.host().file("Pipfile"))
            .with_file("/app/Pipfile.lock", client.host().file("Pipfile.lock"))
            .with_workdir("/app")
            .with_exec(["sh", "-c", "pipenv requirements > requirements.txt"])
        )
        requirements_content = await app_reqs.file("requirements.txt").contents()

        runner = (
            client.container()
            .from_("python:3.12-slim")
            # Single layer for system dependencies
            .with_exec([
                "sh", "-c",
                "apt-get update && \
                apt-get install -y --no-install-recommends libprotobuf-dev && \
                apt-get clean && \
                rm -rf /var/lib/apt/lists/*"
            ])
            .with_exec(["pip", "install", "--no-cache-dir", "protobuf==5.28.3"])
            .with_exec(["pip", "install", "--no-cache-dir", "gunicorn==23.0.0"])
            .with_directory("/root/.cache/tokenizer_models", models_dir)
            .with_directory("/usr/local/lib/python3.12/site-packages/en_core_web_sm", spacy_model)
            .with_directory("/usr/local/lib/python3.12/site-packages/en_core_web_sm-3.8.0.dist-info", model_dist)
            .with_directory("/app/src", client.host().directory("src"))
            .with_workdir("/app")
            .with_file("requirements.txt",
                       client.directory().with_new_file("requirements.txt", contents=requirements_content).file(
                           "requirements.txt"))
            .with_exec(["pip", "install", "--no-cache-dir", "-r", "requirements.txt"])
            .with_env_variable("FLASK_ENV", "production")
            .with_env_variable("PYTHONPATH", "src")
            .with_exposed_port(5000)
            .with_entrypoint(["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "src.web_app:app"])
        )

        with open("VERSION", "r") as version_file:
            version = version_file.read().strip()
        repo = "nestharus/difwordcount"

        await runner.publish(f"{repo}:{version}")
        await runner.publish(f"{repo}:latest")

anyio.run(main)