package main

import (
    "dagger.io/dagger"
    "universe.dagger.io/docker"
)

dagger.#Plan & {
    client: filesystem: {
        "./": read: {
            contents: dagger.#FS
            exclude: [
                "dagger.cue",
                ".git",
                ".pytest_cache",
                "__pycache__",
                "*.pyc"
            ]
        }
    }

    actions: {
        build: docker.#Build & {
            steps: [
                docker.#Pull & {
                    source: "python:3.11-slim"
                },
                docker.#Set & {
                    config: workdir: "/app"
                },
                docker.#Run & {
                    command: {
                        name: "pip"
                        args: ["install", "pipenv"]
                    }
                },
                docker.#Copy & {
                    contents: client.filesystem."./".read.contents
                    dest: "."
                },
                docker.#Run & {
                    command: {
                        name: "pipenv"
                        args: ["install", "--deploy"]
                    }
                },
                docker.#Run & {
                    command: {
                        name: "pipenv"
                        args: ["run", "setup"]
                    }
                },
                docker.#Set & {
                    config: {
                        env: FLASK_ENV: "production"
                        expose: ["5000"]
                        entrypoint: ["pipenv", "run", "start"]
                    }
                }
            ]
        }

        run: docker.#Run & {
            input: build.output
            command: {
                name: "pipenv"
                args: ["run", "start"]
            }
            ports: publish: "5000": "5000"
        }
    }
}
