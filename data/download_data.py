#!/usr/bin/python

# Martin Kersner, m.kersner@gmail.com
# 2015/11/04

# Modified script from @ramhiser https://gist.github.com/ramhiser/4121260#file-download-data-py

# Download all necessary data for specified Kaggle competition.
# User has to be registered using normal account, therefore accounts registered
# under Google, Yahoo or Facebook are not compatible with this script.

import os
import re
import getpass
import requests

# TODO  automatically accept competition rules

invalid_login = "The username or password provided is incorrect."

def download_from_kaggle(data, credential):
    verified_credentials = False

    for data_url in data:
        filename = get_filename(data_url)

        r = requests.get(data_url) # attempt to download file is rejected
        r = requests.post(r.url, data = kaggle_credentials, stream = True) # login to Kaggle and retrieve data

        if not verified_credentials:
            verified_credentials = verify_credentials(r.text, invalid_login)
        
        download_file(r, filename)

def download_file(request, filename):
    print "Downloading " + filename, 

    with open(filename, 'w') as f:
        for chunk in request.iter_content(chunk_size = 512 * 1024): # Reads 512KB at a time into memory
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

    print "done!"

def get_filename(path):
    return os.path.split(path)[-1]

def verify_credentials(html_text, invalid_login):
    re_result = re.search(invalid_login, html_text, flags=0)
    if (re_result != None):
       print invalid_login
       exit()
    
    return True

def retrive_kaggle_data_urls(competition_data_path, kaggle_website):
    competition_data_path = competition_data_path.strip()

    # html page
    r = requests.get(competition_data_path)
    if (r.status_code != 200):
        wrong_competition_data_path()

    # modifications of given competition path
    regex_sub = re.compile('%s'%kaggle_website)
    link2search = re.sub(regex_sub, "", competition_data_path)

    if (len(link2search) == len(competition_data_path)):
        wrong_competition_data_path

    link2search = link2search[0:-4] + "download/" 
    regex_findall = re.compile('(%s.*?)\"'%link2search)
    if (len(link2search) == 0):
        wrong_competition_data_path

    partial_data_urls = re.findall(regex_findall, r.text, flags=0)
    full_data_urls = [kaggle_website + s for s in partial_data_urls]

    return full_data_urls

def wrong_competition_data_path():
    print "Given competition data page is wrong."
    exit()

def accept_competition_rules():
    pass

def main():
    kaggle_website = "https://www.kaggle.com"

    # e.g. https://www.kaggle.com/c/street-view-getting-started-with-julia/data
    competition_data_path = raw_input("Enter full url of Kaggle competition data page: ")

    #accept_competition_rules()
    
    kaggle_data_urls = retrive_kaggle_data_urls(competition_data_path, kaggle_website)

    # Kaggle username and password 
    username = raw_input("Enter username: ")
    password = getpass.getpass("Enter password: ")

    kaggle_credentials = {"UserName": str(username), "Password": str(password)}

    download_from_kaggle(kaggle_data_urls, kaggle_credentials)

if __name__ == "__main__":
    main()
