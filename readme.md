## Abstract
While text classification can classify tweets, assessing whether a tweet is related to an ongoing flood event or not, based on its text, remains difficult. Inclusion of contextual hydrological information could improve the performance of such algorithms. Here, a multilingual multimodal neural network is designed that can effectively use both textual and hydrological information. The classification data was obtained from Twitter using flood-related keywords in English, French, Spanish and Indonesian. Subsequently, hydrological information was extracted from a global precipitation dataset based on the tweet's timestamp and locations mentioned in its text. Three experiments were performed analyzing precision, recall and F1-scores while comparing a neural network that uses hydrological information against a neural network that does not. Results showed that F1-scores improved significantly across all experiments. Most notably, when optimizing for precision the neural network with hydrological information could achieve a precision of 0.91 while the neural network without hydrological information failed to effectively optimize. Moreover, this study shows that including hydrological information can assist in the translation of the classification algorithm to unseen languages.

## Cite as
de Bruijn, Jens A., et al. "Improving the classification of flood tweets with contextual hydrological information in a multimodal neural network." Computers & Geosciences (2020): 104485.

## How to run
1. Setup
    - Install Python 3.6 and all modules in requirements.txt.
    - Install PostgreSQL (tested with 10.1) and `POSTGRESQL_HOST`, `POSTGRESQL_PORT`, `POSTGRESQL_USER` and `POSTGRESQL_PASSWORD` in *config.py*.
2. Hydrological input data
    - Register at https://sharaku.eorc.jaxa.jp/GSMaP/registration.html and obtain username and password for ftp-server.
    - Fill out `GSMaP_USERNAME` and `GSMaP_PASSWORD` in *config.py*
    - Run *1. download_GSMaP_hourly.py*. This file will download hourly GSMaP data to the folder *classification/data/GSMaP/raw*. This will take a while.
    - Run *2. process_rainfall.py*. This will compile a NetCDF4-file out of the downloaded files.
    - Download the HydroBasins dataset for all contintents from https://www.hydrosheds.org/downloads. Select "Standard (Without lakes)" level 9.
    - Unpack the HydroBasins dataset  to *data/hybas* (e.g., *data/hybas/hybas_af_lev09_v1c.shp*)
    - Download hydrosheds_connectivity.zip from https://doi.org/10.5281/zenodo.1015799
    - Unpack the data to *data/hydrorivers* (e.g., *data/hydrorivers/afriv/rapid_connect_HSmsp_tst.csv*)
    - Download *riverPolylines.zip* from https://doi.org/10.5281/zenodo.1015799
    - Unpack the data to the corresponding folders in data/hydrorivers (e.g., *data/hydrorivers/afriv/afriv.shp*)
    - Run *3. create_river_and_basin_maps.py*
    - Run *4. find_basin_indices.py*
3. Textual input data
    - Download the MUSE repository (https://github.com/facebookresearch/MUSE) to source folder (e.g., *MUSE/supervised.py*)
    - Run *5. get_word_embeddings.py*. This will download word embeddings from Facebook and create multilingual word embeddings using MUSE. This process is a lot faster when Faiss is installed.
4. General
    - Obtain a `TWITTER_CONSUMER_KEY`, `TWITTER_CONSUMER_SECRET`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_TOKEN_SECRET` from Twitter by registering as a developer at Twitter (https://developer.twitter.com/) and set keys in *config.py*.
    - Run *6. hydrate.py*. This obtain text, date and language for tweets in the labelled data and place the hydrated data in data/labeled_data_hydrated.csv.
    - Run *7. create_input_data.py* This will read *data/labeled_data_hydrated.csv* and outputs a pickle with word embeddings and hydrological data per tweet. The data is placed in *data/input*.
    - Run *8. experiments.py*. This will run all experiments.
    - Run *9. analyze_results.py*. The results of the experiments are placed in the *results* folder.