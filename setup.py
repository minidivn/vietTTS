from setuptools import setup

__version__ = '1.0.0b'
url = 'https://github.com/ntt123/vietTTS'

install_requires = [
    'dm-haiku @ git+https://github.com/deepmind/dm-haiku', 
    'einops', 
    'gdown'
    'jax', 
    'jaxlib', 
    'librosa', 
    'optax', 
    'tabulate', 
    'textgrid', 
    'tqdm', 
]
setup_requires = []
tests_require = []

setup(
    name='vietTTS',
    version=__version__,
    description='A vietnamese text-to-speech engine',
    author='ntt123',
    url=url,
    keywords=['text-to-speech', 'tts', 'deep-learning', 'dm-haiku', 'jax', 'vietnamese', 'speech-synthesis'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=['vietTTS'],
    python_requires='>=3.6',
)
