from imdb import IMDb

imdbInstance = IMDb()
movieId = "0120815"
movieData = imdbInstance.get_movie(movieId)

print(len(movieData['plot']))

print(movieData['plot'][2])