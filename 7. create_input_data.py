import pickle
import re
import numpy as np
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
import tensorflow as tf
import os
import netCDF4 as nc
import numpy as np
import xarray as xr

from datetime import timedelta, datetime
from scipy.stats import percentileofscore

from postgresql import PostgreSQL
from data_helpers import load_embeddings_and_mapping, get_word_vector
from config import (
    SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    SAMPLE_SETS
)

pg = PostgreSQL('classification')


class BasinHydrology():
    def __init__(self, name_rainfall_dataset, data, basin_id, correct_rainfall=True, discard_below_or_equal_to_value=-1):
        self.name_rainfall_dataset = name_rainfall_dataset
        self.precipitation = data
        self.basin_id = basin_id
        self.correct_rainfall = correct_rainfall
        self.discard_below_or_equal_to_value = discard_below_or_equal_to_value

        self.delta = timedelta(hours=1)
        
        self.data = {}

        self.start_date = self.numpy64_todatetime(self.precipitation['time'][0].values)

    def numpy64_todatetime(self, np64):
        assert isinstance(np64, np.datetime64)
        return datetime.utcfromtimestamp(np64.astype('O') / 1e9)

    def find_upstream_basins(self, id, level, max_recursion=None):
        upstream_basins = []
        if max_recursion is None or max_recursion > 0:
            if max_recursion is not None:
                max_recursion -= 1
            pg.cur.execute(f"""
                SELECT id FROM subbasins_{level} WHERE downstream = '{id}'
            """)
            for upstream_id, in pg.cur.fetchall():  # if none are found the recursive function is never called thus is a stop condition
                upstream_basins.append(upstream_id)
                upstream_basins.extend(self.find_upstream_basins(upstream_id, level, max_recursion))
        return upstream_basins


    def find_upstream_rivers_by_travel_time(self, river_id, lag_time_hours=None, _downstream_basin=None):
        upstream_rivers = []
        if _downstream_basin is None:
            pg.cur.execute("""
                SELECT subbasin_9 FROM hydrorivers WHERE id = %s
            """, (river_id, ))
            _downstream_basin = pg.cur.fetchone()[0]
        if lag_time_hours is None or lag_time_hours > 0:
            if lag_time_hours is not None:
                pg.cur.execute("""
                    SELECT subbasin_9, propagation_time FROM hydrorivers WHERE id = %s
                """, (river_id, ))
                current_basin, propagation_time_days = pg.cur.fetchone()
                if current_basin != _downstream_basin:
                    lag_time_hours -= propagation_time_days * 24  # days -> hours
            pg.cur.execute(f"""
                SELECT id FROM hydrorivers WHERE downstream = %s
            """, (river_id, ))
            for upstream_id, in pg.cur.fetchall():  # if none are found the recursive function is never called thus is a stop condition
                upstream_rivers.append(upstream_id)
                upstream_rivers.extend(self.find_upstream_rivers_by_travel_time(
                    upstream_id, lag_time_hours, _downstream_basin=_downstream_basin))
        return upstream_rivers

    def get_river_travel_time(self, upstream_id, downstream_id):
        total_propagation_time = 0
        while upstream_id != downstream_id:
            pg.cur.execute("""
                SELECT propagation_time, downstream FROM hydrorivers WHERE id = %s
            """, (upstream_id, ))
            propagation_time, new_id = pg.cur.fetchone()
            total_propagation_time += propagation_time
            upstream_id = new_id
        return total_propagation_time * 24

    def get_basins_for_rivers(self, river_ids):
        pg.cur.execute("""
            SELECT DISTINCT subbasin_9 FROM hydrorivers WHERE id IN %s
        """, (tuple(river_ids), ))
        return [basin_id for basin_id, in pg.cur.fetchall()]

    def find_upstream_basins_by_travel_time(self, basin_id, lag_time_hours=None):
        pg.cur.execute("""
            WITH river_segments AS (
                SELECT
                    id,
                    downstream
                FROM
                    hydrorivers
                WHERE
                subbasin_9 = %s
            )
            SELECT
                id
            FROM river_segments
            WHERE downstream NOT IN (
                SELECT id
                FROM river_segments
            ) OR downstream IS NULL
        """, (basin_id, ))

        upstream_rivers = set()
        for river_id, in pg.cur.fetchall():
            upstream_rivers.update(self.find_upstream_rivers_by_travel_time(river_id, lag_time_hours=lag_time_hours))
        if upstream_rivers:
            basins = self.get_basins_for_rivers(upstream_rivers)
            try:
                basins.remove(basin_id)
            except ValueError:
                pass
            return basins
        else:
            return []

    def create_timeline(self, precipitation, accumulation_time):
        assert accumulation_time != 0
        cumsum = np.cumsum(precipitation)
        cumsum = np.insert(cumsum, 0, 0)
        timeline = (cumsum[accumulation_time:] - cumsum[:-accumulation_time]) / accumulation_time
        return timeline

    def get_cell_precipitation(self, x, y):
        return self.precipitation[
            {'lat': y, 'lon': x}
        ]['precipitation'].values

    def get_hydrology_for_basin(self, basinid):
        pg.cur.execute("""
            SELECT indices
            FROM """ + self.name_rainfall_dataset.lower() + """_basin_indices
            WHERE idx = %s
        """, (basinid, ))
        indices = pg.cur.fetchone()[0]
        basin_precipitation = []
        factors = []
        for cell_indices in indices:
            cell_precipitation = self.get_cell_precipitation(
                cell_indices['x'],
                cell_indices['y']
            )
            if self.name_rainfall_dataset == 'PERSIANN':
                cell_precipitation = cell_precipitation / 100  # unit is 100 * mm/hr
            elif self.name_rainfall_dataset == 'GSMaP':
                pass  # unit is mm/r
            else:
                raise ValueError
            if self.correct_rainfall:
                factor = cell_indices['area_basin'] / cell_indices['area_cell']
            else:
                factor = 1
            basin_precipitation.append(cell_precipitation)
            factors.append(factor)
        return (basin_precipitation, factors)
    
    def set_data(self, precipitation, factors, upstream_hours):
        self.data[upstream_hours] = (precipitation, factors)

    def set_basin_hydrology(self):
        precipitation, factors = self.get_hydrology_for_basin(self.basin_id)
        self.set_data(precipitation, factors, upstream_hours=0)

    def set_upstream_basins_hydrology(self, upstream_hours):
        basin_ids = self.find_upstream_basins_by_travel_time(
            self.basin_id, lag_time_hours=upstream_hours
        ) + [self.basin_id]

        assert len(basin_ids) == len(set(basin_ids))

        upstream_factors = []
        upstream_basins_precipitation = []
        for basin_id in basin_ids:
            basin_precipitation, factors = self.get_hydrology_for_basin(basin_id)
            upstream_basins_precipitation.extend(basin_precipitation)
            upstream_factors.extend(factors)

        self.set_data(upstream_basins_precipitation, upstream_factors, upstream_hours=upstream_hours)

    def get_index_date(self, date, accumulation_time):
        assert accumulation_time >= 1
        start_date = self.start_date + (accumulation_time - 1) * self.delta  # obtain start time for index
        index = (date - start_date) // self.delta
        if index < 0:
            raise ValueError("Cannot go that far back")
        else:
            return index

    def discard_below_or_equal_to(self, timeline):
        if self.discard_below_or_equal_to != -1:
            timeline = timeline[timeline > self.discard_below_or_equal_to_value]
        return timeline

    def calculate_percentile_of_value(self, timeline, value, accumulation_time):
        # if accumulation_time != 1:
        #     import pdb; pdb.set_trace()
        if value < self.discard_below_or_equal_to_value:
            value = 0
        timeline = self.discard_below_or_equal_to(timeline)
        # mean, std = np.mean(timeline), np.std(timeline)
        # if value <= mean:
        #     return 0
        # else:
        #     return (value - mean) / std
        if self.discard_below_or_equal_to_value != -1:
            timeline = timeline[timeline != 0]
        # if timeline.size == 0:
        #     if value > 0:
        #         return 1
        #     else:
        #         return 0
        # print(value, np.percentile(timeline, 98), value / np.percentile(timeline, 98))
        # return value / np.percentile(timeline, 98)
        return percentileofscore(timeline, value, kind='strict')

    def calculate_percentile(self, timeline, threshold):
        timeline = self.discard_below_or_equal_to(timeline)
        return np.percentile(timeline, threshold)        

    def get_timeline(self, upstream_hours, accumulation_time):
        precipitation, factors = self.data[upstream_hours]
        if self.correct_rainfall:
            precipitation_ = np.zeros_like(precipitation[0])
            for i in range(len(precipitation)):
                precipitation_ += (factors[i] * precipitation[i])
            precipitation = precipitation_ / sum(factors)
        else:
            precipitation = np.sum(precipitation, axis=0)
        if accumulation_time > 1:
            timeline = self.create_timeline(precipitation, accumulation_time)
        else:
            timeline = precipitation
        return timeline

    def get_percentile(self, date, upstream_hours, accumulation_time, time_period=1):
        timeline = self.get_timeline(upstream_hours, accumulation_time)
        if time_period > 1:
            upper_index = self.get_index_date(date, accumulation_time) + 1
            lower_index = self.get_index_date(date - timedelta(hours=time_period - 1), accumulation_time)
            values = timeline[lower_index:upper_index]
            value = max(values)
        else:
            index = self.get_index_date(date, accumulation_time)
            value = timeline[index]
        return self.calculate_percentile_of_value(timeline, value, accumulation_time)

    def get_percentile_above_threshold(self, date, upstream_hours, accumulation_time, threshold, time_period=1):
        precipitation, factors = self.data[upstream_hours]
        if time_period > 1:
            upper_index = self.get_index_date(date, accumulation_time) + 1
            lower_index = self.get_index_date(date - timedelta(hours=time_period - 1), accumulation_time)
        else:
            index = self.get_index_date(date, accumulation_time)
        n_cells = 0
        for factor, cell_precipitation in zip(factors, precipitation):
            if time_period > 1:
                values = cell_precipitation[lower_index:upper_index]
                value = max(values)
            else:
                value = cell_precipitation[index]
            percentile = self.calculate_percentile(cell_precipitation, threshold)
            if value > percentile:
                n_cells += factor
        return n_cells / sum(factors)


class DataCreator:
    def __init__(self, settings):
        self.settings = settings

    def clean_text(self, ID, text, language_code, locations, user_replacements={
        'en': 'user',
        'nl': 'gebruiker',
        'de': 'benutzer',
        'id': 'pengguna',
        'es': 'usuario',
        'fr': 'utilisateur',
        'it': 'utente',
        'pl': 'użytkownik',
        'pt': 'usuário',
        'tr': 'kullanıcı',
        'tl': 'user',
    }, location_replacements={
        "country": "brazil",
        "adm1": "florida",
        "other": "amsterdam"
    }):
        text = text.lower()

        # replace usernames with "username" in respective language
        usernames = re.findall(r"(?:^|\s)@([A-Za-z0-9_]+)", text)
        if usernames:
            for username in usernames:
                text = text.replace(username, user_replacements[language_code])

        urls = re.findall(r"(\bhttps?://t.co/[a-zA-Z0-9]*\b)", text)
        if urls:
            for url in urls:
                text = text.replace(url, "")

        if self.settings['replace_locations']:
            locations = locations.split(';')
            for location in locations:
                loc_type, toponym = location.split(':')
                if loc_type == 'country':
                    text = text.replace(toponym, location_replacements['country'])
                elif loc_type == 'adm1':
                    text = text.replace(toponym, location_replacements['adm1'])
                else:
                    text = text.replace(toponym, location_replacements['other'])
        return text

    def get_hydrology(self, doc_date, basin_hydrology):
        hydrology = []
        hydrology_labels = []
        for area_considered in ('local', 'local+upstream'):
            for accumulation_time in (1, 3, 24, 5 * 24):
                if area_considered == 'local+upstream' and accumulation_time in (1, 3):
                    continue
                for time_period in (1, 24, 72):
                    if area_considered == 'local':
                        upstream_hours = 0
                    else:
                        upstream_hours = accumulation_time
                    hydrology.append(
                        basin_hydrology.get_percentile(
                            doc_date,
                            upstream_hours=upstream_hours,
                            accumulation_time=accumulation_time,
                            time_period=time_period
                        )
                    )
                    hydrology_labels.append(f'{accumulation_time}h_max{time_period}h_{area_considered}_percentile')

                    for percentile_threshold in (90, 95, 98):
                        hydrology.append(
                            basin_hydrology.get_percentile_above_threshold(
                                doc_date,
                                upstream_hours=upstream_hours,
                                accumulation_time=accumulation_time,
                                threshold=percentile_threshold,
                                time_period=time_period
                            )
                        )
                        hydrology_labels.append(f'{accumulation_time}h_max{time_period}h_{area_considered}_above_{percentile_threshold}')

        assert None not in hydrology
        assert not np.isnan(np.array(hydrology)).any()
        assert not np.isinf(np.array(hydrology)).any()
        assert len(hydrology) == len(hydrology_labels)
        # assert all(100 >= value >= 0 for value in hydrology)

        return hydrology, hydrology_labels

    def get_doc_data(self, ID, date, text, language_code, locations, context_function=None, context_data=None):

        text = self.clean_text(ID, text, language_code, locations)

        if context_function:
            context, context_labels = context_function(date, context_data)
            return text, context, context_labels
        else:
            return text

    def load_rainfall_data(self, start_date, end_date):
        files = [
            os.path.join('data', self.settings['rainfall_dataset'], f'1hr_sum_{year}.nc')
            for year in range(start_date.year, end_date.year + 1)
        ]
        ds = xr.open_mfdataset(files)
        precipitation = ds.sel(time=slice(start_date, end_date))
        return precipitation
        
    def process(self, data, res_file, include_labels=False, include_context=False, summarize=False):
        res_folder = os.path.join('data', 'input')
        try:
            os.makedirs(res_folder)
        except OSError:
            pass
        res_file = os.path.join(res_folder, res_file)
        pickle_file = res_file + '.pickle'
        if os.path.exists(pickle_file):
            print(pickle_file, "already exists")
            return None
        
        ids = defaultdict(list)
        texts = defaultdict(list)
        if include_labels:
            labels = defaultdict(list)
        
        if include_context:
            context = defaultdict(list)
            if include_labels:
                context_per_class = {
                    label: []
                    for label in (0, 1)
                }
            if include_context == 'hydrology':
                tweets_by_basin = defaultdict(list)
                for subbasin, *tweet_info in data:
                    assert subbasin.startswith('s-')  # check if indeed subbasin
                    assert len(tweet_info) == 6
                    tweets_by_basin[subbasin].append(tweet_info)

                rainfall_data = self.load_rainfall_data(
                    datetime(2009, 1, 1),
                    datetime(2018, 12, 31, 23)
                )

                n_subbasins = len(tweets_by_basin)
                for i, (subbasin, tweets) in enumerate(tweets_by_basin.items(), start=1):
                    print(f"subbasin {i}/{n_subbasins}")
                    
                    basin_hydrology = BasinHydrology(
                        self.settings['rainfall_dataset'],
                        rainfall_data,
                        subbasin,
                        correct_rainfall=self.settings['correct_rainfall'],
                        discard_below_or_equal_to_value=self.settings['discard_below_or_equal_to_value']
                    )
                    basin_hydrology.set_basin_hydrology()
                    basin_hydrology.set_upstream_basins_hydrology(24)
                    basin_hydrology.set_upstream_basins_hydrology(5 * 24)

                    n_tweets = len(tweets)
                    for i, (ID, date, text, language_code, locations, label) in enumerate(tweets, start=1):
                        print(f'{i}/{n_tweets}', end='\r')
                        text, tweet_context, context_labels = self.get_doc_data(ID, date, text, language_code, locations=locations, context_data=basin_hydrology, context_function=self.get_hydrology)

                        # fill arrays for sample (text, rainfall, labels)
                        ids[language_code].append(ID)
                        texts[language_code].append(text)
                        context[language_code].append(tweet_context)
                        if include_labels:
                            labels[language_code].append(label)

                        # Some summarization to calculate rainfall statistics
                        if include_labels:
                            context_per_class[label].append(tweet_context)
            else:                    
                raise NotImplementedError
            # Summarize
            if include_labels and summarize:
                summary = pd.DataFrame(
                    [
                        np.average(context_class, axis=0)
                        for context_class in context_per_class.values()
                    ],
                    index=context_per_class.keys(),
                    columns=context_labels
                )
                summary.to_excel(res_file + '.xlsx')
                print(summary)
        else:
            for ID, date, text, language_code, label in data:
                text = self.get_doc_data(ID, date, text, language_code)
                # fill arrays for sample (text, rainfall, labels)
                ids[language_code].append(ID)
                texts[language_code].append(text)
                if include_labels:
                    labels[language_code].append(label)

        total_num_words = 0
        all_text_sequences = []
        all_context = []
        all_ids = []
        all_languages = []
        all_sentences = []
        if include_labels:
            all_labels = []
        embedding_matrices = []

        for language_code, language_texts in texts.items():
            tokenizer = tf.keras.preprocessing.text.Tokenizer()
            tokenizer.fit_on_texts(language_texts)

            embeddings, mapping = load_embeddings_and_mapping(language_code)

            num_words_lang = len(tokenizer.word_index) + 1
            embedding_matrix_lang = np.zeros((num_words_lang, EMBEDDING_DIM))
            
            for word, i in tokenizer.word_index.items():
                embedding_vector = get_word_vector(embeddings, mapping, word)
                embedding_matrix_lang[i] = embedding_vector

            embedding_matrices.append(embedding_matrix_lang)

            text_sequences = tokenizer.texts_to_sequences(language_texts)
            text_sequences = [
                [
                    index + total_num_words
                    for index in sequence
                ]
                for sequence in text_sequences
            ]

            text_sequences = tf.keras.preprocessing.sequence.pad_sequences(text_sequences, maxlen=SEQUENCE_LENGTH)
            all_text_sequences.extend(text_sequences)    
            all_ids.extend(ids[language_code] )     
            
            if include_context:
                context_lang = np.array(context[language_code], dtype=np.float32)
                all_context.extend(context_lang)
            if include_labels:
                all_labels.extend(labels[language_code])

            total_num_words += num_words_lang

            all_languages.extend([language_code] * len(language_texts))
            all_sentences.extend(language_texts)

        embedding_matrix = np.vstack(embedding_matrices)

        if include_labels:
            all_labels = np.asarray(all_labels, dtype=np.int32)


        data = {
            'ids': all_ids,
            'sentences': all_sentences,
            'text_sequences': np.array(all_text_sequences, dtype=np.int32),
            'languages': all_languages
        }

        if include_context:
            data['context'] = np.vstack(all_context)
        if include_labels:
            data['labels'] = np.array(all_labels, dtype=np.int32)

        res = {
            'data': data,
            'embedding_matrix': embedding_matrix,
        }
        if include_context:
            res['context_labels'] = context_labels
        
        with open(pickle_file, 'wb') as p:
            pickle.dump(res, p)

    def analyze_labelled_data(self):
        labeled_data = pd.read_csv('data/labeled_data_hydrated.csv')
        for sample_set in SAMPLE_SETS:
            tweets = []
            for _, tweet in labeled_data[labeled_data['language'] == sample_set].iterrows():
                tweets.append((
                    's-' + str(tweet['subbasin']),
                    't-' + str(tweet['ID']),
                    datetime.strptime(tweet['date'], '%a %b %d %H:%M:%S +0000 %Y'),
                    tweet['text'],
                    tweet['language'],
                    tweet['locations'],
                    tweet['label']
                ))
            if tweets:
                res_file = f"data_{sample_set}_correct_rainfall_{self.settings['correct_rainfall']}_discard_below_or_equal_to_value_{self.settings['discard_below_or_equal_to_value']}_{self.settings['rainfall_dataset']}"
                self.process(tweets, res_file=res_file, include_labels=True, include_context='hydrology')

    def analyze_tweets_subbasin(self, subbasin, languages=None):
        from db.elastic import Elastic
        es = Elastic()
        query = {
            'query': {
                'term': {
                    'locations.subbasin_ids_9': subbasin
                }
            }, 
            'sort': {
                'date': 'asc'
            }
        }

        data = []
        tweets = es.scroll_through(index='floods_all', body=query, source=False)
        for tweet in tweets:
            detailed_locations = [loc for loc in tweet['locations'] if loc['type'] in ('town', 'adm5', 'adm4', 'adm3', 'landmark')]
            if len(detailed_locations) != 1:
                continue

            detailed_location = detailed_locations[0]
            if subbasin not in detailed_location['subbasin_ids_9']:
                continue

            if detailed_location['score'] < .2:
                continue

            tweet_lang = tweet['source']['lang']
            if languages and tweet_lang not in languages:
                continue

            data.append((
                subbasin,
                tweet['id'],
                tweet['date'],
                tweet['text'],
                tweet_lang,
                None
            ))
        self.process(data, res_file=subbasin, include_context='hydrology')


if __name__ == '__main__':
    settings = {
        "rainfall_dataset": 'GSMaP',
        "discard_below_or_equal_to_value": 0,
        "correct_rainfall": True,
        "replace_locations": True
    }
    data_creator = DataCreator(settings)
    data_creator.analyze_labelled_data()