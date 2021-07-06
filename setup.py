import setuptools

with open('./README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('./requirements.txt', 'r') as rfh:
    lib_flag = False
    requirements = []

    for line in rfh.readlines():
        if line.startswith('#') and 'lib' in line:
            lib_flag = True
        elif lib_flag and line:
            requirements.append(line)
        elif not line:
            break

setuptools.setup(
    name='webfish-tools',
    version='0.0.1',
    author='Lincoln Ombelets',
    author_email='lombelets@caltech.edu',
    description='The utilities and backend for the webfish UI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/CaiGroup/web-ui',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    #package_dir={'': '.'},
    packages=['lib'],
    python_requires='>=3.9',
    install_requires=requirements
)
