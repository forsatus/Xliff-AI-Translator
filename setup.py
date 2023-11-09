from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='xliff-ai-translator',
    version='1.0',
    packages=find_packages(),
    description=('AI-driven solution for efficient XLIFF/XLF file translation '
                 ', bridging the gap between languages for localized content.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='sutasrof',
    author_email='mario.chauvet@icloud.com',
    url='https://github.com/sutasrof/Xliff-AI-Translator',
    install_requires=[
        'transformers[torch]>=4.34.1',
        'sentencepiece>=0.1.99',
        'lxml>=4.9.3'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'xliff_ai_translator=xliff_ai_translator.main:main',
        ],
    },
)
