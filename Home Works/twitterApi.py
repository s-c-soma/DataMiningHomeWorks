import twitter


api = twitter.Api(consumer_key='pM8VhaCuNyMr7tuPiJGSACyr0',
                  consumer_secret='H4D3NtRpSEXmR7aseowaoFedMLt1I7BEyUI5M20q7rH2kqM5hk',
                  access_token_key='1180201895551328257-OhsK5pMcFNxiyJWUlyCwfp6ArHc5Lj',
                  access_token_secret='rN8S32KUULpewDWmxvkrefG3r6JyeKuyoNbxWXIIEgAyZ')


obama = api.GetUserTimeline(screen_name='BlueHatsSjsu')
print(obama)

#Message = "My first post via twitter api-subarna"
#api.PostUpdate(Message)

#api.DestroyStatus("1182903529838534657")
#print(Message)
