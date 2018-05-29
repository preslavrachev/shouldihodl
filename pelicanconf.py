#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
from datetime import datetime
import json

AUTHOR = 'Preslav Rachev'
SITENAME = 'Should I HODL?'
SITEURL = ''

PATH = 'content'

THEME = 'theme/'

TIMEZONE = 'Europe/Berlin'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Pelican', 'http://getpelican.com/'),
         ('Python.org', 'http://python.org/'),
         ('Jinja2', 'http://jinja.pocoo.org/'),
         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = False

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

DECISION_DATA = {}
with open('decision.json', 'r') as fp:
    DECISION_DATA = json.load(fp)
    DECISION_DATA['last_update'] = datetime.fromtimestamp(DECISION_DATA['timestamp'])

print(DECISION_DATA)