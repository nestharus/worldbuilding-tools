```
docker pull nestharus/difwordcount:latest
docker stop difwordcount
docker remove difwordcount
docker run -d -p 80:5000 --name difwordcount nestharus/difwordcount:latest
```

https://difcheck.guildies.gg/

```
pipenv run start
dagger run python ci/publish.py
dagger run python ci/run.py
```