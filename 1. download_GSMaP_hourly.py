from datetime import datetime, timedelta
import multiprocessing as mp
import pysftp
import paramiko
import gzip
import time
import numpy as np
from os import path, remove, rename, makedirs
from pathlib import Path
from operator import le, lt


from config import GSMaP_USERNAME, GSMaP_PASSWORD


local_folder = path.join('data', 'GSMaP', 'raw')

try:
    makedirs(local_folder)
except OSError:
    pass


def daterange(start_date, end_date, delta, ranges=False, include_last=False, UTC=False, timedelta=timedelta):
    if UTC:
        start_date = start_date.replace(tzinfo=pytz.UTC)
        end_date = end_date.replace(tzinfo=pytz.UTC)
    if not isinstance(delta, timedelta):
        delta = timedelta(seconds=int(delta))
    if include_last:
        sign = le
    else:
        sign = lt
    while sign(start_date, end_date):
        if ranges:
            yield start_date, start_date + delta
        else:
            yield start_date
        start_date += delta


data_type = np.dtype('<f4')  # Little-endian 4byte (32bit) float
lons = np.arange(0.05, 360, 0.1)
lats = np.arange(59.95, -59.95 - 0.1, -0.1)
data_lon_size = len(lons)
data_lat_size = len(lats)


def connect_jaxa():
    global jaxa_server
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None
    host = 'hokusai.eorc.jaxa.jp'
    while True:
        try:
            jaxa_server = pysftp.Connection(host, username=GSMaP_USERNAME, password=GSMaP_PASSWORD, cnopts=cnopts)
            break
        except pysftp.exceptions.ConnectionException:
            print("retrying")
            time.sleep(30)


def GSMaP_downloader(q):
    connect_jaxa()
    while True:
        dt = q.pop()
        f = dt.strftime("gsmap_nrt.%Y%m%d.%H00.dat.gz")

        fp_unchecked = path.join(local_folder, '_' + f)
        fp_checked = path.join(local_folder, f)

        print('checking', fp_unchecked)
        if path.exists(fp_unchecked):
            print('exists')
            try:
                with gzip.open(fp_unchecked, 'rb') as t:
                    t.read()
            except EOFError:
                remove(fp_unchecked)
            else:
                rename(fp_unchecked, fp_checked)
                continue
    
        remote_fp = path.join(dt.strftime('/realtime/archive/%Y/%m/%d/'), f)
        while True:
            try:
                print("trying", remote_fp)
                print("to", fp_unchecked)
                jaxa_server.get(remote_fp, localpath=fp_unchecked)
            except FileNotFoundError:
                print(remote_fp, 'not found - going to sleep for a bit (180s)')
                time.sleep(180)
            except (paramiko.ssh_exception.SSHException, OSError):
                print("reconnecting jaxa")
                connect_jaxa()
            else:
                try:
                    with gzip.open(fp_unchecked, 'rb') as g:
                        np.frombuffer(g.read(), dtype=data_type).reshape(data_lat_size, data_lon_size)
                except (EOFError, ValueError):
                    remove(fp_unchecked)
                else:
                    rename(fp_unchecked, fp_checked)
                    print('renamed')
                    break
        print(remote_fp)
        # check if empty
        if not q:
            print('appended', dt + timedelta(hours=1))
            q.append(dt + timedelta(hours=1))



if __name__ == '__main__':
    GSMaP_missing = []

    now = datetime.utcnow()
    start = datetime(2009, 1, 1)
    for dt in daterange(start, now, timedelta(hours=1)):
        f = dt.strftime("gsmap_nrt.%Y%m%d.%H00.dat.gz")
        if not path.exists(
            path.join(local_folder, f)
        ):
            GSMaP_missing.append(dt)
    GSMaP_missing = GSMaP_missing[::-1]

    GSMaP_downloader_p = mp.Process(target=GSMaP_downloader, args=(GSMaP_missing, ))
    GSMaP_downloader_p.start()
