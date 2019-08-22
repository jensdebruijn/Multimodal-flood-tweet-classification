from copy import copy
import numpy as np
import os
import sys
import netCDF4
from netCDF4 import date2num
import gzip
from datetime import timedelta, datetime
import subprocess
from pathlib import Path
import pytz

from operator import le, lt

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



def create_nc(start, end, folder, nc_path):
    if RAINFALL_TYPE == 'PERSIANN':
        data_type = np.dtype('>i2')  # Little-endian 4byte (32bit) float
        lons = np.arange(-179.98, 180, 0.04)
        lats = np.arange(59.98, -60, -0.04)
    elif RAINFALL_TYPE == 'GSMaP':
        data_type = np.dtype('<f4')
        lons = np.arange(-179.95, 180, 0.1)
        lats = np.arange(59.95, -59.95 - 0.1, -0.1)
    else:
        raise ValueError
    
    data_lon_size = len(lons)
    data_lat_size = len(lats)
    if not os.path.exists(nc_path):

        # Create netcdf file
        ncfile = netCDF4.Dataset(nc_path, 'w')

        # create a dimensions
        ncfile.createDimension('time', 0)
        ncfile.createDimension('lat', data_lat_size)
        ncfile.createDimension('lon', data_lon_size)
        
        lat = ncfile.createVariable('lat', 'f4', ('lat',))
        lat.standard_name = 'latitude'
        lat.units = 'degrees_north'
        lat.axis = "Y"
        lat[:] = lats
        
        lon = ncfile.createVariable('lon', 'f4', ('lon',))
        lon.standard_name = 'longitude'
        lon.units = 'degrees_east'
        lon.axis = "X"
        lon[:] = lons
        
        times = ncfile.createVariable('time', 'f4', ('time',))
        times.standard_name = 'time'
        times.long_name = 'time'
        times.units = 'hours since 1970-01-01 00:00:00'
        times.calendar = 'gregorian'
        precip = ncfile.createVariable(
            'precipitation',
            datatype='i2',
            dimensions=('time', 'lat', 'lon'),
            chunksizes=(30, 20, 20),
            zlib=True,
            complevel=1,  # ignored if zlib is False
            contiguous=False,  # do not neccesarily store contiguous on disk
            fill_value=-9999
        )
        precip.standard_name = 'precipitation'
        precip.units = 'mm'
        timestep = 0

    else:
        ncfile = netCDF4.Dataset(nc_path, 'a')
        times = ncfile.variables['time']

        timestep = len(times)
        start = start + timedelta(hours=timestep)

        precip = ncfile.variables['precipitation']

    precip.set_var_chunk_cache(size=14_000_000_000, nelems=100_000_000, preemption=0)
    t0 = datetime.utcnow()
    for timestep, dt in enumerate(daterange(start, end, timedelta(hours=1)), start=timestep):
        start_of_year = datetime(dt.year, 1, 1)
        days_since_start = (dt - start_of_year).days + 1
        
        if RAINFALL_TYPE == 'PERSIANN':
            fn = f"rgccs1h{str(dt.year)[2:]}{str(days_since_start).zfill(3)}{str(dt.hour).zfill(2)}.bin.gz"
        elif RAINFALL_TYPE == 'GSMaP':
            fn = 'raw/gsmap_nrt.{}{:02d}{:02d}.{:02d}00.dat.gz'.format(dt.year, dt.month, dt.day, dt.hour)

        fp = os.path.join(folder, fn)

        if os.path.exists(fp):
            print("timestep: {} - {}".format(timestep, dt))

            with gzip.open(fp, 'rb') as g:
                try:
                    rainfall_raw = np.frombuffer(g.read(), dtype=data_type).reshape(data_lat_size, data_lon_size)
                except (OSError, ValueError):
                    print('OSError')
                    rainfall = np.full((data_lat_size, data_lon_size), np.nan)
                else:
                    rainfall_raw = copy(rainfall_raw)
                    
                    split_col = data_lon_size // 2
                    
                    rainfall = np.zeros((data_lat_size, data_lon_size))
                    rainfall[:, split_col:] = rainfall_raw[:, :split_col]
                    rainfall[:, :split_col] = rainfall_raw[:, split_col:]

                    # rainfall is never negative
                    rainfall[rainfall < 0] = 0
                
                precip[timestep, :, :] = rainfall
                times[timestep] = date2num(dt, units=times.units, calendar=times.calendar)
        else:
            print(f'{fp} does not exist')
            sys.exit()
        t1 = datetime.utcnow()
        print(t1 - t0)
        t0 = t1

        if not timestep % 30:
            ncfile.sync()

    ncfile.close()


if __name__ == '__main__':
    RAINFALL_TYPE = 'GSMaP'
    base_path = os.path.join('data', RAINFALL_TYPE)
    for year in list(range(2009, 2019)):
        hourly_nc = os.path.join(base_path, f"1hr_sum_{year}.nc")
        create_nc(
            datetime(year, 1, 1),
            datetime(year + 1, 1, 1),
            base_path,
            hourly_nc
        )