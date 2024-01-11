# Notes

Here are some things I need to put into CookieCuttter

- vscode devcontainer extensions
- vscode user settings
- Poetry 1.7
- install of `poetry-plugin-export` as a self plugin
- Dockerfile supply arguments via `--build-arg` during Github Action

```bash
> cat .dvc/config 
[core]
    remote = {env.repository_name}
['remote "{env.repository_name}"']
    url = s3://with-context-sagemaker/datasets/{env.repository_name}/
```
