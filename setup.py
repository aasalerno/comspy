from setuptools import setup, find_packages

setup(name='comspy',
      version = '0.1.1',
      description = "A pythonic Compressed Sensing Library",
      url = "https://github.com/aasalerno/comspy",
      author = "Anthony Salerno",
      author_email = "anthony.salerno@sickkids.ca",
      license = "BSD",
      packages = find_packages(exclude = ['data', 'logs', 'examples', 'temp', 'tests']),
      install_requires = ["numpy","scipy","matplotlib","PyWavelets"],
      scripts = [],
#      test_suite="test"
)


