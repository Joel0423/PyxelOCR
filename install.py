import subprocess
from nltk import download

cmd = ['python','-m','textblob.download_corpora']
subprocess.run(cmd)
print("TextBlob corpora installed")

download('punkt_tab')
# Download NLTK words list
download('words')