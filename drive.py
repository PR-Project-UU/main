import io
import pickle
import os.path
from typing import Callable

from logging import getLogger

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive']

class Drive:
    log = getLogger('drive')
    service = None

    def __init__(self):
        '''Creates a Drive object for accessing drive files'''
        creds = None

        # Load saved credentials if available
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)

        # If the credentials have expired or are unvailable, refresh them and let the user sign in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port = 0)
            with open('token.pickle', 'wb') as token:
                token.dump(creds, token)

        self.service = build('drive', 'v3', credentials=creds)

    def download(self, name: str, extension: str, save_path='./', delete: bool = False) -> bool:
        self.log.info('Looking to download "%s"', name)

        log = self.log.getChild('dl:"%s"' % name)
        page_token = None
        results = []

        # Make sure the save path ends in a slash
        if save_path[-1] != '/':
            save_path += '/'

        # Make sure the extension starts with a dot
        if extension[0] != '.':
            extension = '.' + extension

        # Make sure the folder to save to exists
        if not os.path.exists(save_path):
            log.error('Path to save to ("%s") does not exist', save_path)
            return False

        # Make sure the file to save doesn't already exist
        if os.path.exists(save_path + name + extension):
            log.error('File already exists at path "%s"', save_path + name)
            return False

        # Find the file in the drive
        while True:
            response = self.service.files().list(q="name='%s'" % name, spaces='drive', fields='nextPageToken, files(id, name)', pageToken=page_token).execute()
            page_token = response.get('nextPageToken', None)

            for file in response.get('files', []):
                results.push((file.get('name'), file.get('id')))

            if page_token is None:
                break

        # Make sure we got any results
        if len(results) == 0:
            log.info('Found no files with name "%s"', name)
            return False

        # Download the file(s)
        for i, (file_name, file_id) in enumerate(results):
            log.info('Downloading file %d of %d with name %s', i, len(results), file_name)

            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh , request)
            done = False

            while not done:
                _, done = downloader.next_chunk()

            # Save the file
            local_file_name = save_path + name + extension if i == 0 else save_path + name + ('(%s)' % i) + extension

            with open(local_file_name, 'wb') as local_file:
                local_file.write(fh.getbuffer())

            if delete:
                log.info('Deleting downloaded file "%s" from Drive', file_name)
                self.service.files().delete(file_id).execute()

        return True
