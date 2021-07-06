import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="webfish-tools",
    version="0.0.1",
    author="Lincoln Ombelets",
    author_email="lombelets@caltech.edu",
    description="The utilities and backend for the webfish UI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CaiGroup/web-ui",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    package_dir={"": "lib"},
    packages=setuptools.find_packages(where="lib"),
    python_requires=">=3.9",
)
