import tweepy


def auth():
    """
    Se connecte à l'api twitter
    """
    # PS: les accès ont été re-généré au moment de rendre ce repository publique :)
    consumer_key = "X"
    consumer_secret = "X"

    access_token = "X"
    access_token_secret = "X"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    return api,auth
