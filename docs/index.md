---
sd_hide_title: true
hide-toc: true
---

# AutoGluon-Assistant

::::::{div} landing-title
:style: "padding: 0.1rem 0.5rem 0.6rem 0; background-image: linear-gradient(315deg, #438ff9 0%, #3977B9 74%); clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem)); -webkit-clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem));"

::::{grid}
:reverse:
:gutter: 2 3 3 3
:margin: 4 4 1 2

:::{grid-item}
:columns: 12 4 4 4

```{image} ./_static/autogluon-s.png
:width: 200px
:class: sd-m-auto sd-animate-grow50-rot20
```
:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-text-white sd-fs-3

AutoGluon-Assistant: Fast and Accurate ML in 0 Lines of Code

:::
::::

::::::

AutoGluon Assistant is a multi-agent system that automates end-to-end multimodal machine learning or deep learning workflows by transforming raw multimodal data into high-quality ML solutions with zero human intervention. Leveraging specialized perception agents, dual-memory modules, and iterative code generation, it handles diverse data formats while maintaining high success rates across complex ML tasks.

Currently, AutoGluon-Cloud uses [MLZero](<https://arxiv.org/abs/2505.13941>) as the backend.

## {octicon}`package` Installation

![](https://img.shields.io/pypi/pyversions/autogluon.cloud)
![](https://img.shields.io/pypi/v/autogluon.cloud.svg)
![](https://img.shields.io/pypi/dm/autogluon.cloud)

```bash
pip install autogluon.assistant  # You don't need to install autogluon itself locally
```

## {octicon}`rocket` Quick Start

:::{dropdown} CLI
:animate: fade-in-slide-down
:open:
:color: primary

```python
mlzero -i ./data
```
:::


:::{dropdown} WebUI
:animate: fade-in-slide-down
:color: primary

```python
mlzero-frontend
```
:::


```{toctree}
---
caption: Tutorials
maxdepth: 3
hidden:
---

Assistant <tutorials/index>
```