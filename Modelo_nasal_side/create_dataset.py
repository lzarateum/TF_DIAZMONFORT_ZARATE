import os
import json
import jsbeautifier
import cv2
import numpy as np
import base64
import io
from PIL import Image

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Define custom progress bar
progress_bar = Progress(
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)






directory = 'originals_mkx/'
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.mkx')]
files.sort()


with progress_bar as p:
    for c in p.track(range(len(files))):
        name_mkx = files[c]
        print(f'{name_mkx} {c+1}/{len(files)}')
        with open(os.path.join(directory, name_mkx), 'br') as file:
            data = file.read()
        data = json.loads(data)
        img = data['image_data']
        img = base64.b64decode(img)
        img = Image.open(io.BytesIO(img))
        img = np.array(img)
        
        if name_mkx.find('_OD') != -1:
            cv2.imwrite('./dataset/nasalder/'+name_mkx[:-4] + '.jpg', img)
        else:
            cv2.imwrite('./dataset/nasalizq/'+name_mkx[:-4] + '.jpg', img)

        

        #        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        
