from imdb import IMDb

# create an instance of the IMDb class
ia = IMDb()

# get a movie and print its director(s)
the_matrix = ia.get_movie('8108198')
plot=the_matrix['plot']
print(plot[0])
print("\n\n\n\n\n\n")
print(plot[1])

# # Synopsis
# plot=the_matrix['synopsis']

# for director in the_matrix['directors']:
#     print(director['name'])

# # show all information that are currently available for a movie
# print(sorted(the_matrix.keys()))

# # show all information sets that can be fetched for a movie
# print(ia.get_movie_infoset())

# # update a Movie object with more information
# ia.update(the_matrix, ['technical'])
# # show which keys were added by the information set
# print(the_matrix.infoset2keys['technical'])
# # print one of the new keys
# print(the_matrix.get('tech'))