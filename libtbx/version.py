from __future__ import absolute_import, division, print_function

import os
import subprocess
import sys
import time

# =============================================================================
def create_version_files(git_repo='cctbx_project', basename='cctbx_version',
                         version=None, setup_template=None):
  '''
  Function for creating the files containing the version. This
  function is called by bootstrap.py after downloading the git
  repository. The development version is the date of the commit and
  the commit information from "git describe". Files containing an
  official release version can created by providing the version as an
  argument

  Parameters
  ----------
  git_repo: str
    The git repository to be versioned. This is the directory name in
    "modules" (e.g. cctbx_project)
  basename: str
    The basename for the filenames. It is also the name of the defintion
    in the C++ header. The ".txt" and ".h" extensions will be added to
    the filenames.
  version: str
    If set, this argument is used as the version
  setup_template: str
    A template for the setup.py file. There should be a {version} field.

  Returns
  -------
  filenames: str
    A tuple containing the three filenames. The first is the plain text
    file, the second is a C++ header file, and the last is a setup.py
    file. These files are located in the git repository. The plain text
    and header cab be copied to the build directory in libtbx_refresh.py.
  '''

  if version is None:
    tagged = False  # True for tagged release

    # create {y}.{m}.dev{d}+{n}.{h} formatted version
    y = None  # year
    m = None  # month
    d = None  # day
    n = None  # number of commits since last tag
    h = None  # g + hash of commit

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', git_repo)
    if not os.path.isdir(path):
      raise RuntimeError('The {path} directory does not exist.'.format(path=path))

    try:
      t = subprocess.check_output(['git', 'log', '-1', '--pretty=%ci'], cwd=path).decode('utf8')
      t = t.split()[0].split('-')
      y = int(t[0])
      m = int(t[1])
      d = int(t[2])
    except subprocess.CalledProcessError:
      t = time.localtime()
      y = t.tm_year
      m = t.tm_mon
      d = t.tm_mday
    try:
      output = subprocess.check_output(['git', 'describe'], cwd=path).decode('utf8')
      output = output.split('-')
      if len(output) == 1:  # tagged release does not have -
        tagged = True
        version = output[0][1:]  # remove first v
      else:
        n = int(output[-2])
        h = output[-1].strip()
    except subprocess.CalledProcessError:
      pass

    if not tagged:
      version = '{y}.{m}.dev{d}'.format(y=y, m=m, d=d)

      # add latest commit information as local version
      if n is not None and h is not None:
        version += '+{n}.{h}'.format(n=n, h=h)
      else:
        version += '+unknown'

  # write plain text
  txt_filename = os.path.join(path, basename + '.txt')
  with open(txt_filename, 'w') as f:
    f.write(version)

  # write C++ header
  header_template = '''\
// {basename} version header
// This file is automatically generated

#ifndef {basename}_H
#define {basename}_H

#define {basename} "{version}"

#endif
'''
  h_filename = os.path.join(path, basename + '.h')
  with open(h_filename, 'w') as f:
    f.write(header_template.format(basename=basename.upper(), version=version))

  # write setup.py
  if setup_template is None:
    setup_template = '''\
from setuptools import setup
setup(
    name='cctbx-base',
    version='{version}',
    url='https://github.com/cctbx/cctbx_project',
    description='The Computational Crystallography Toolbox (cctbx) is being developed as the open source component of the Phenix system. The goal of the Phenix project is to advance automation of macromolecular structure determination. Phenix depends on the cctbx, but not vice versa. This hierarchical approach enforces a clean design as a reusable library. The cctbx is therefore also useful for small-molecule crystallography and even general scientific applications.',
    author='CCTBX developers',
    author_email='cctbx@cci.lbl.gov',
    maintainer='CCTBX developers',
    maintainer_email='cctbx@cci.lbl.gov',
    license='BSD-3-Clause-LBNL AND BSD-3-Clause AND BSL-1.0 AND LGPL-2.0-only AND LGPL-2.1-only AND LGPL-3.0-only AND MIT AND LGPL-2.0-or-later WITH WxWindows-exception-3.1'
)
'''

  setup_filename = os.path.join(path, 'setup.py')
  with open(setup_filename, 'w') as f:
    f.write(setup_template.format(version=version))

  return (txt_filename, h_filename, setup_filename)

# -----------------------------------------------------------------------------
def get_version(filename='cctbx_version.txt', fail_with_none=False):
  '''
  Function for returning the version of the current installation
  The file containing the version is manually created for official
  releases and created by bootstrap.py for development releases. The
  version follows calendar versioning, so in the event that the file
  cannot be read, a version based on the current date can be returned.

  Parameters
  ----------
  filename: str
    The filename of the file containing the version number. This can
    be a full path. Otherwise, the directory is assumed to be the build
    directory.
  fail_with_none: bool
    If set, any failure to read the version file will return None.
    Otherwise, the current date is used to generate the version.

  Returns
  -------
  str or None
  '''
  import libtbx.load_env

  version = None
  path = filename
  if not os.path.isabs(path):
    path = os.path.join(abs(libtbx.env.build_path), filename)

  try:
    with open(path) as f:
      version = f.read().strip()
  except IOError:
    if fail_with_none:
      return None

  if version is None:
    t = time.localtime()
    version = '{y}.{m}.dev{d}+unknown'.format(y=t.tm_year, m=t.tm_mon, d=t.tm_mday)

  return version

# =============================================================================
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(
    description='Command for generating files containing version information.')
  parser.add_argument(
    '--git-repo',
    help='The name of the git repository in the "modules" directory.',
    default='cctbx_project')
  parser.add_argument(
    '--basename',
    help='The base name for the version filenames.',
    default='cctbx_version')
  parser.add_argument(
    '--version',
    help='An explicit version to be set.',
    default=None)

  namespace = parser.parse_args(sys.argv[1:])

  filenames = create_version_files(
    git_repo=namespace.git_repo,
    basename=namespace.basename,
    version=namespace.version)

  print('Writing files containing version information for {git_repo}'.\
    format(git_repo=namespace.git_repo))
  print('='*79)
  for filename in filenames:
    print('Wrote {filename}'.format(filename=filename))
  print('='*79)

# =============================================================================
# end
