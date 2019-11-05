from links import *
from selenium import webdriver
import time
import urllib3

chrome = webdriver.Chrome('chromedriver.exe')

def extract_link(prefix):
    hrefs = [a.get_attribute('href') for a in chrome.find_elements_by_css_selector('a')]
    hrefs = [href for href in hrefs if prefix in href]
    if len(hrefs) > 0: return hrefs[0]
    else: return None

def get_download_links(links, prefix):
    outputs = []
    for link in links:
        print(link)
        resp = chrome.get('https://9xbuddy.com/process?url='+link)
        url = None
        while url is None:
            time.sleep(0.1)
            url = extract_link(prefix)
        outputs.append(url)
    return outputs

links = get_download_links(ipl_links, IPL_PREFIX) + get_download_links(cwc_links, CWC_PREFIX)

http = urllib3.PoolManager()
for out in links:
    r = http.request('GET', out)
    data = r.data
    fname = 'raw/' + out.split('=')[-1] + '.mp4'
    with open(fname, 'wb') as fobj:
        fobj.write(data)        