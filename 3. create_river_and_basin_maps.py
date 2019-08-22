from postgresql import PostgreSQL
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from osgeo import ogr

pg = PostgreSQL('classification')


def export_shapefile(shapefile, fields, kind='json', force_multipolygon=False):
    shapefile = ogr.Open(shapefile)
    layer = shapefile.GetLayer(0)
    output = []
    for i in range(layer.GetFeatureCount()):
        row = []
        feature = layer.GetFeature(i)
        for field in fields:
            row.append(feature.GetField(field))
        geometry = feature.GetGeometryRef()
        if force_multipolygon and geometry.GetGeometryType() == ogr.wkbPolygon:
            geometry = ogr.ForceToMultiPolygon(geometry)
        if kind in ('json', 'geojson'):
            geometry = geometry.ExportToJson()
        elif kind == 'wkt':
            geometry = geometry.ExportToWkt()
        else:
            raise NotImplementedError
        row.append(geometry)
        output.append(row)
    return output

def create_subbasin_map(level):
    if not pg.table_exists(f'subbasins_{level}'):
        print(f"Creating subbasins map level {level}")
        pg.cur.execute(f"""
            CREATE TABLE subbasins_{level} (
                id VARCHAR PRIMARY KEY,
                downstream VARCHAR,
                geom GEOMETRY(Multipolygon, 4326)
            )
        """)
        hybas_basins = []
        for continent in ['af', 'eu', 'ar', 'as', 'au', 'na', 'sa', 'si']:
            f = f'hybas/hybas_{continent}_lev{level:02d}_v1c.shp'
            for basin in export_shapefile(
                os.path.join(
                    'data', f
                ),
                ['HYBAS_ID', 'NEXT_DOWN'],
                kind='wkt'
            ):
                basin[0] = 's-' + str(basin[0])
                basin[1] = 's-' + str(basin[1]) if basin[1] != 0 else None
                hybas_basins.append(basin)

        pg.execute_values(
            f"INSERT INTO subbasins_{level} (id, downstream, geom) VALUES %s",
            hybas_basins,
            template="""(
            %s,
            %s,
            ST_Multi(
                ST_CollectionExtract(ST_MakeValid(ST_GeometryFromText(%s, 4326)), 3)
            )
            )""",
            page_size=1000
        )

        pg.cur.execute(f"""
            CREATE INDEX
            IF NOT EXISTS subbasins_{level}_geometry_idx
            ON subbasins_{level}
            USING GIST (geom)
        """)

        pg.cur.execute(f"""
            CREATE INDEX
            IF NOT EXISTS subbasins_{level}_downstream_idx
            ON subbasins_{level} (downstream)
        """)
        pg.conn.commit()

def create_hydrorivers_map():
    if not pg.table_exists('hydrorivers'):
        print('Creating hydrobasins map')
        pg.cur.execute("""
            CREATE TABLE hydrorivers (
                id INT PRIMARY KEY,
                downstream INT,
                upstream_cells INT,
                propagation_time REAL,
                geom Geometry(LineString, 4326)
            )
        """)
        continents = ['ca', 'af', 'eu', 'as', 'au', 'na', 'sa']

        n_so_far = 0
        for continent in continents:
            folder = os.path.join('data', 'hydrorivers', f'{continent}riv')
            gdf = gpd.GeoDataFrame.from_file(os.path.join(folder, f'{continent}riv.shp')) 

            connections = os.path.join(folder, 'rapid_connect_HSmsp_tst.csv')
            connections = pd.read_csv(connections, usecols=[0, 1], names=[
                                    'id', 'downstream_id'])
            connections['downstream_id'] = np.rint(connections['downstream_id'])
            connections['downstream_id'] = connections['downstream_id'].replace(
                0, np.nan)
            connections['downstream_id'] += n_so_far
            connections = connections.set_index('id')['downstream_id'].to_dict()

            gdf['id'] = np.rint(gdf['ARCID']).astype(int)

            gdf['downstream_id'] = gdf['id'].map(connections)
            gdf['id'] += n_so_far

            for _, row in gdf.iterrows():
                pg.cur.execute("""
                    INSERT INTO hydrorivers (id, downstream, upstream_cells, propagation_time, geom) VALUES (%s, %s, %s, %s, ST_SetSRID(ST_GeomFromText(%s), 4326))
                """, (row['id'], row['downstream_id'], row['UP_CELLS'], row['PROPTIME_D'], row['geometry'].wkt))

            pg.conn.commit()
            n_so_far += len(gdf)

        print("Finding subbasins for river segments")
        pg.cur.execute("""
            ALTER TABLE hydrorivers ADD COLUMN subbasin_9 VARCHAR
        """)
        pg.cur.execute("""
            UPDATE hydrorivers
            SET subbasin_9 = subbasins_9.id
            FROM subbasins_9
            WHERE ST_Contains(subbasins_9.geom, ST_LineInterpolatePoint(hydrorivers.geom, 0.5))
        """)

        pg.cur.execute("""
            CREATE INDEX
            IF NOT EXISTS hydrorivers_downstream_idx
            ON hydrorivers (downstream)
        """)

        pg.conn.commit()


if __name__ == '__main__':
    create_subbasin_map(9)
    create_hydrorivers_map()