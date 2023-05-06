from setuptools import setup, find_packages

setup(
  name = 'nebula-loss',
  packages = find_packages(exclude=[]),
  version = '0.3.0',
  license='MIT',
  description = '1 Loss Function to rule them all!',
  author = 'Agora',
  author_email = 'kye@apac.ai',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/kyegomez/nebula',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'jax',
    'loss ffunction',
    "Multi-Modality AI"
  ],
  install_requires=[
    'torch',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)