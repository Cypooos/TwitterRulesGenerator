import tweepy

# From your app settings page
CONSUMER_KEY = "04ITuPOxz3TsmGU5ijBvZeLED"
CONSUMER_SECRET = "zV1OslaCuWaVJkGOYssgLIc4BEN5Dr8PUDlb2SgGPfDrBupJQ3"

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.secure = True
auth_url = auth.get_authorization_url()

print('Please authorize: ' + auth_url)

verifier = input('PIN: ').strip()

auth.get_access_token(verifier)

print("ACCESS_KEY = '%s'" % auth.access_token.key)
print("ACCESS_SECRET = '%s'" % auth.access_token.secret)