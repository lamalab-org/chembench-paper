<p align="center">
<a href="https://github.com/showyourwork/showyourwork">
<img width = "450" src="https://raw.githubusercontent.com/showyourwork/.github/main/images/showyourwork.png" alt="showyourwork"/>
</a>
<br>
<br>
<a href="https://github.com/lamalab-org/chembench-paper/actions/workflows/build.yml">
<img src="https://github.com/lamalab-org/chembench-paper/actions/workflows/build.yml/badge.svg?branch=main" alt="Article status"/>
</a>
<!-- <a href="https://github.com/lamalab-org/chembench-paper/raw/main-pdf/arxiv.tar.gz">
<img src="https://img.shields.io/badge/article-tarball-blue.svg?style=flat" alt="Article tarball"/>
</a> -->
<a href="https://github.com/lamalab-org/chembench-paper/raw/main/ms.pdf">
<img src="https://img.shields.io/badge/article-pdf-blue.svg?style=flat" alt="Read the article"/>
</a>
</p>

An open source scientific article created using the [showyourwork](https://github.com/showyourwork/showyourwork) workflow.
# chembench-paper

Build the article using the following command:

```bash
showyourwork build --conda-frontend=mamba --rerun-incomplete
```

Note that you need to have your Zenodo tokens exported (`export ZENODO_TOKEN=<your token>` and `export SANDBOX_TOKEN=<your token>`).
For the first rules, you also need to have a Github personal access token exported (`export GH_PAT=<your token>`).
For the LLM-parsing rule, you need your Anthropic API key exported (`export ANTHROPIC_API_KEY=<your key>`).

## Useful command

If stuff stops working, you can try to reset the cache

```bash
showyourwork clean --force --deep
```

(You might need to manually delete the downloaded `.git` folder).

If you are confused on why stuff is not working, check the logs in `.showyourwork/logs`.
