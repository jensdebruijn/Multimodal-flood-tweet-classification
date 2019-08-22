import subprocess
import os
import gzip
import shutil

from config import SAMPLE_SETS as LANGUAGES
LANGUAGES = ['en', 'id']

data_folder = os.path.join('MUSE', 'data')
try:
    os.makedirs(data_folder)
except OSError:
    pass
fastText_folder = os.path.join('data', 'fastText')
try:
    os.makedirs(fastText_folder)
except OSError:
    pass

for language in LANGUAGES:
    local_path_zip = os.path.join(data_folder, f"wiki.{language}.vec.gz")
    if not os.path.exists(local_path_zip):
        cmd = f'curl -Lo "{local_path_zip}" https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{language}.300.vec.gz'
        print(cmd)
        subprocess.call(cmd, shell=True)

    output_vec = os.path.join(data_folder, f'wiki.{language}.vec')
    if not os.path.exists(output_vec):
        print('Unpacking', local_path_zip)
        with gzip.open(local_path_zip, 'rb') as gz, open(output_vec, 'wb') as output:
            shutil.copyfileobj(gz, output)

    local_path_zip = os.path.join(fastText_folder, f"wiki.{language}.bin.gz")
    if not os.path.exists(local_path_zip):
        cmd = f'curl -Lo "{local_path_zip}" https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{language}.300.bin.gz'
        print(cmd)
        subprocess.call(cmd, shell=True)

    output_vec = os.path.join(fastText_folder, f'wiki.{language}.bin')
    if not os.path.exists(output_vec):
        print('Unpacking', local_path_zip)
        with gzip.open(local_path_zip, 'rb') as gz, open(output_vec, 'wb') as output:
            shutil.copyfileobj(gz, output)

crosslingual_folder = os.path.join(data_folder, 'crosslingual')
try:
    os.mkdir(crosslingual_folder)
except OSError:
    pass
dictionary_folder = os.path.join(data_folder, 'crosslingual', 'dictionaries')
try:
    os.mkdir(dictionary_folder)
except OSError:
    pass

for language in LANGUAGES:
    if language != 'sr':
        local_path = os.path.join(dictionary_folder, f"{language}-en.0-5000.txt")
        if not os.path.exists(local_path):
            cmd = f'curl -Lo "{local_path}" https://dl.fbaipublicfiles.com/arrival/dictionaries/{language}-en.0-5000.txt'
            print(cmd)
            subprocess.call(cmd, shell=True)
        local_path = os.path.join(dictionary_folder, f"{language}-en.5000-6500.txt")
        if not os.path.exists(local_path):
            cmd = f'curl -Lo "{local_path}" https://dl.fbaipublicfiles.com/arrival/dictionaries/{language}-en.5000-6500.txt'
            print(cmd)
            subprocess.call(cmd, shell=True)

dumped_dir = os.path.join('MUSE', 'dumped', 'classification')
try:
    os.makedirs(dumped_dir)
except OSError:
    pass

for language in LANGUAGES:
    if language == 'en':
        continue
    else:
        if not os.path.exists(os.path.join(fastText_folder, language, f'best_mapping.pth')):
            cmd = f'python MUSE/supervised.py --src_lang {language} --tgt_lang en --src_emb MUSE/data/wiki.{language}.vec --tgt_emb MUSE/data/wiki.en.vec --n_refinement 5 --dico_train default --cuda false --exp_name classification --exp_id {language} --export "" --n_refinement 5'
            print(cmd)
            subprocess.call(cmd, shell=True)
            try:
                os.makedirs(os.path.join(fastText_folder, language))
            except OSError:
                pass
            os.rename(
                os.path.join(dumped_dir, language, 'best_mapping.pth'),
                os.path.join(fastText_folder, f'best_mapping.{language}.pth')
            )
