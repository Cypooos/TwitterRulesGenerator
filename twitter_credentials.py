import tweepy


def auth():
    """
    se connecte à l'api twitter
    """
    consumer_key = "04ITuPOxz3TsmGU5ijBvZeLED"
    consumer_secret = "zV1OslaCuWaVJkGOYssgLIc4BEN5Dr8PUDlb2SgGPfDrBupJQ3"

    access_token = "1373256143120777216-zhTc0c5jecSWfMJLn63uqS5g9v2q7t"
    access_token_secret = "aeVQnsoJrdP7w8STeDXuYto8SuLt5PGIkfC4ppBwgK8ep"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    return api,auth