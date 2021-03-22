from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy

import twitter_credentials

import AI.generate_rule as g_r

from time import sleep


api,auth = twitter_credentials.auth()
last_one = None


while True:
    rule_raw = g_r.model_generator.one_rule(255,last_one)
    rule_utf = str(rule_raw.numpy(), encoding='utf-8', errors = 'ignore')
    rule = rule_utf.split("\r\n")[1].replace(":newline:","\n")
    print("New Rule in 20 seconds:",rule)

    sleep(20)
    t = api.update_status(rule)

    
    # api.upda

    sleep(580)