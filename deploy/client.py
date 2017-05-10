import requests
from StringIO import StringIO
from PIL import Image
import shutil
from io import BytesIO

url = "http://diggerdu.ml:42513/"
files = {'style': open('test.png', 'rb'), 'content': open('test.png', 'rb')}
response = requests.post(url, files=files)
a = Image.open(BytesIO(response.content))
a.save('a.png')
print response.cookies
print response.request

