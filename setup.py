from setuptools import find_packages, setup


def parse_requirements_file(path):
    requirements = []
    with open(path) as requirements_file:
        import re

        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL dependencies should be handled."""
            m = re.match(
                r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git",
                req,
            )
            if m is None:
                return req
            elif m.group("name") == "tango":
                # There is no way to specify extras on the pip command line when doing `pip install <url>`, so
                # there is no way to set up an equivalency between the `pip install` syntax and the `setup.py`
                # syntax. So we just hard-code it here in the case of tango.
                return f"ai2-tango[all] @ {req}"
            elif m.group("name") == "lm-evaluation-harness":
                return f"lm-eval @ {req}"
            elif m.group("name") == "promptsource":
                return f"promptsource @ {req}"
            else:
                return f"{m.group('name')} @ {req}"

        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            req, *comment = line.split("#")
            req = fix_url_dependencies(req.strip())
            requirements.append(req)
    return requirements


# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import `cached_path` whilst setting up.
VERSION = {}  # type: ignore
with open("catwalk/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="ai2-catwalk",
    version=VERSION["VERSION"],
    description="A library for evaluating language models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    url="https://github.com/allenai/catwalk",
    author="Allen Institute for Artificial Intelligence",
    author_email="contact@allenai.org",
    license="Apache",
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "test_fixtures",
            "test_fixtures.*",
        ],
    ),
    package_data={
        "catwalk": [
            "py.typed",
            "dependencies/promptsource/templates/*/*.yaml",
            "dependencies/promptsource/templates/*/*/*.yaml"
        ]
    },
    install_requires=parse_requirements_file("requirements.txt"),
    extras_require={"dev": parse_requirements_file("dev-requirements.txt")},
    python_requires=">=3.8.0",
)
