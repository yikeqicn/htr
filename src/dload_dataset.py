import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--data_root', default=os.path.join(os.getenv('HOME'), 'datasets'), type=str, help='root where datasets are stored')
# parser.add_argument('--source_url', default='https://www.dropbox.com/sh/24lkzoxfktgdl20/AABzQYYglGF0o1FR5lhHxDSqa?dl=0', type=str, help='url for data')
# parser.add_argument('--filename', default='cifar-100-binary', type=str, help='file/folder name for data')
# parser.add_argument('--source_url_pretrain', default='https://www.dropbox.com/sh/axypx1vrdbay2x1/AABA7DdqeTNt5lu5g9IhXzAaa?dl=0', type=str, help='url for pretrain')
# parser.add_argument('--filename_pretrain', default='17', type=str, help='file/folder name for pretrain')
args = parser.parse_args()

def maybe_download(source_url, filename, target_directory, filetype='folder', force=False):
  """Download the data from some website, unless it's already here."""
  if source_url==None or filename==None: return
  if target_directory==None: target_directory = os.getcwd()
  filepath = os.path.join(target_directory, filename)
  if os.path.exists(filepath) and not force:
    print(filepath+' already exists, skipping download')
  else:
    if not os.path.exists(target_directory):
      os.system('mkdir -p '+target_directory)
    if filetype=='folder':
      os.system('curl -L '+source_url+' > '+filename+'.zip')
      os.system('unzip -o '+filename+'.zip'+' -d '+filepath)
      os.system('rm '+filename+'.zip')
    elif filetype=='tar':
      os.system('curl -o '+filepath+'.tar '+source_url)
      os.system('tar xzvf '+filepath+'.tar --directory '+target_directory)
      os.system('rm '+filepath+'.tar')
    else:
      os.system('wget -O '+filepath+' '+source_url)

if args.dataset=='cifar100':
  source_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'
  filename = 'cifar-100-binary'
elif args.dataset=='cifar10':
  source_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
  filename = 'cifar-10-batches-bin'
elif args.dataset=='iam':
  source_url = 'https://www.dropbox.com/sh/tdd0784neuv9ysh/AABm3gxtjQIZ2R9WZ-XR9Kpra?dl=0'
  filename = 'iam_handwriting'

# download data
print('source url = '+source_url)
print('filename = '+filename)
maybe_download(source_url=source_url,
                     filename=filename,
                     target_directory=args.data_root,
                     filetype='folder')

