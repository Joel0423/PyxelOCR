import subprocess
from nltk import download

cmd = ['python3','-m','textblob.download_corpora']
subprocess.run(cmd)
print("TextBlob corpora installed")

download('punkt_tab')