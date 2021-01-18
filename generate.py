from drive import Drive
from logging import getLogger
from os import path
import pandas as pd
import pickle
from satellite import Capture
from sys import exc_info
from time import sleep
from typing import List, Tuple

class Generator():
    available = 3000
    dataframe = None
    download: List[Tuple[str, str]] = []
    drive = Drive()
    log = getLogger('generator')
    status = dict()

    def __init__(self):
        # Check if a status has already been set
        if path.exists('./generate.pickle'):
            with open('./generate.pickle', 'rb') as f:
                self.status = pickle.loads(f.read())

        for city in self.status:
            self.download.extend((name, city) for name in self.status[city]['tdl'])

        self.available = 3000 - len(self.download)
        self.dataframe = pd.read_csv('./meta_features.csv')

    def run(self, save_path: str, delete: bool):
        '''Runs the generating process and catches keyboard interrupts'''
        try:
            for city in self.dataframe['METROREG'].unique():
                downloaded = []

                for i, (download_name, download_city) in enumerate(self.download):
                    if self.drive.download(download_name, '.tif', save_path, delete):
                        downloaded.append(i)
                        self.status[download_city]['tdl'].remove(download_name)

                shift = 0
                for item in sorted(downloaded):
                    del self.download[item - shift]
                    shift += 1

                self.available += len(downloaded)

                if not city in self.status:
                    self.status[city] = {
                        'done': False,
                        'index': 0,
                        'tdl': []
                    }
                    self.log.info('Created status entry for "%s"', city)

                if self.status[city]['done']:
                    self.log.info('Already generated "%s"', city)
                    continue

                self.download.extend([(item, city) for item in self.status[city]['tdl']])
                subset = self.dataframe.loc[self.dataframe['METROREG'] == city]
                start_year = min(subset['TIME'])
                end_year = max(subset['TIME']) + 1
                capture = Capture(subset['latitude'].iloc[0], subset['longitude'].iloc[0], 25, start_year, end_year)
                length = (end_year - start_year) * 25
                
                if self.status[city]['index'] == length and len(self.status[city]['tdl']) == 0:
                    self.status[city]['done'] = True
                    self.log.info('Finished generating and downloading "%s"', city)
                    continue

                for i in range(self.status[city]['index'], length):
                    if self.available <= 0:
                        self.log.warning('At generate limit (available = %s), waiting five minutes', self.available)
                        sleep(300)
                        break

                    capture.generate(i)
                    name = capture.nameFromIndex(i)
                    self.status[city]['index'] += 1
                    self.status[city]['tdl'].append(name)
                    self.download.append((name, city))
                    self.available -= 1

                with open('./generate.pickle', 'wb') as f:
                    f.write(pickle.dumps(self.status))
                    self.log.debug('Written status to "generate.pickle" file')

            self.log.info('Called to generate all cities, waiting for downloads now')
            
            while len(self.download) > 0:
                sleep(30) # Sleep to prevent constant checking

                downloaded = []

                for i, (download_name, download_city) in enumerate(self.download):
                    if self.drive.download(download_name, '.tif', save_path, delete):
                        downloaded.append(i)
                        self.status[download_city]['tdl'].remove(download_name)

                shift = 0
                for item in sorted(downloaded):
                    del self.download[item - shift]
                    shift += 1

        except KeyboardInterrupt:
            self.log.warning('Keyboard interrupt called; saving status.')

            with open('./generate.pickle', 'wb') as f:
                f.write(pickle.dumps(self.status))

            self.log.info('Status saved to "generate.pickle"')
            exit(130)

        except:
            self.log.critical('Encountered an unkown error', exc_info=exc_info()[0])

            with open('./generate.pickle', 'wb') as f:
                f.write(pickle.dumps(self.status))

            self.log.info('Status saved to "generate.pickle"')
            exit(1)

        self.log.info('Finished generating')
